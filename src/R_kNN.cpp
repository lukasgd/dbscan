//----------------------------------------------------------------------
//                  Find the k Nearest Neighbors
// File:                    R_kNNdist.cpp
//----------------------------------------------------------------------
// Copyright (c) 2015 Michael Hahsler. All Rights Reserved.
//
// This software is provided under the provisions of the
// GNU General Public License (GPL) Version 3
// (see: http://www.gnu.org/licenses/gpl-3.0.en.html)

// Note: does not return self-matches!

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <cassert>
#include <algorithm>
#include "ANN/ANN.h"

using namespace Rcpp;

// pre-existing serial implementation

// returns knn + dist
// [[Rcpp::export]]
List kNN_int_serial(NumericMatrix data, int k,
  int type, int bucketSize, int splitRule, double approx) {

  // copy data
  int nrow = data.nrow();
  int ncol = data.ncol();
  ANNpointArray dataPts = annAllocPts(nrow, ncol);
  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol; j++){
      (dataPts[i])[j] = data(i, j);
    }
  }
  //Rprintf("Points copied.\n");

  // create kd-tree (1) or linear search structure (2)
  ANNpointSet* kdTree = NULL;
  if (type==1){
    kdTree = new ANNkd_tree(dataPts, nrow, ncol, bucketSize,
      (ANNsplitRule)  splitRule);
  } else{
    kdTree = new ANNbruteForce(dataPts, nrow, ncol);
  }
  //Rprintf("kd-tree ready. starting DBSCAN.\n");

  NumericMatrix d(nrow, k);
  IntegerMatrix id(nrow, k);

  // Note: the search also returns the point itself (as the first hit)!
  // So we have to look for k+1 points.
  ANNdistArray dists = new ANNdist[k+1];
  ANNidxArray nnIdx = new ANNidx[k+1];

  for (int i=0; i<nrow; i++) {
    if (!(i % 100)) Rcpp::checkUserInterrupt();

    ANNpoint queryPt = dataPts[i];

    kdTree->annkSearch(queryPt, k+1, nnIdx, dists, approx);

    // remove self match
    IntegerVector ids = IntegerVector(nnIdx, nnIdx+k+1);
    LogicalVector take = ids != i;
    ids = ids[take];
    id(i, _) = ids + 1;

    NumericVector ndists = NumericVector(dists, dists+k+1)[take];
    d(i, _) = sqrt(ndists);
  }

  // cleanup
  delete kdTree;
  delete [] dists;
  delete [] nnIdx;
  annDeallocPts(dataPts);
  // annClose(); is now done globally in the package


  // prepare results
  List ret;
  ret["dist"] = d;
  ret["id"] = id;
  ret["k"] = k;
  ret["sort"] = true;
  return ret;
}


// new parallel implementation

struct NearestNeighborFinder : public RcppParallel::Worker {
  
  ANNpointSet* kdTree_;
  const ANNpointArray& dataPts_;
  const int k_;
  const double approx_;
  RcppParallel::RMatrix<double> d_;
  RcppParallel::RMatrix<int> id_;

  NearestNeighborFinder(ANNpointSet* kdTree,
                        const ANNpointArray& dataPts,
                        int k,
                        double approx,
                        NumericMatrix d,
                        IntegerMatrix id) : 
    kdTree_(kdTree), dataPts_(dataPts), k_(k), approx_(approx), d_(d), id_(id) {}

  void operator()(std::size_t begin, std::size_t end) {
    // Note: the search also returns the point itself (as the first hit)!
    // So we have to look for k+1 points.
    std::vector<ANNdist> distances (k_+1);
    std::vector<ANNidx> nnIdx (k_+1);

    for (std::size_t i = begin; i != end; ++i) {
      ANNpoint queryPt = dataPts_[i];

      kdTree_->annkSearch(queryPt, k_+1, nnIdx.data(), distances.data(), approx_);

      // remove self match and increment indices by 1 for R convention
      std::vector<int> other_r_ids;
      std::vector<double>  other_distances;
      for (std::size_t kk = 0; kk < nnIdx.size(); ++kk) {
        if (nnIdx[kk] != (int) i) {
          other_r_ids.push_back(nnIdx[kk] + 1);
          other_distances.push_back(distances[kk]);
        }
      }

      auto id_row = id_.row(i);
      assert(id_.ncol() == other_r_ids.end() - other_r_ids.begin());
      std::copy(other_r_ids.begin(), other_r_ids.end(), id_row.begin());

      auto d_row = d_.row(i);
      assert(d_.ncol() == other_distances.end() - other_distances.begin());
      std::transform(other_distances.begin(), other_distances.end(), d_row.begin(),
                     [](double dist){ return sqrt(dist); });
    }
  }
};

// returns knn + dist
// [[Rcpp::export]]
List kNN_int(NumericMatrix data, int k,
  int type, int bucketSize, int splitRule, double approx) {

  // copy data
  int nrow = data.nrow();
  int ncol = data.ncol();
  ANNpointArray dataPts = annAllocPts(nrow, ncol);
  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol; j++){
      (dataPts[i])[j] = data(i, j);
    }
  }
  //Rprintf("Points copied.\n");

  // create kd-tree (1) or linear search structure (2)
  ANNpointSet* kdTree = NULL;
  if (type==1){
    kdTree = new ANNkd_tree(dataPts, nrow, ncol, bucketSize,
      (ANNsplitRule)  splitRule);
  } else{
    kdTree = new ANNbruteForce(dataPts, nrow, ncol);
  }
  //Rprintf("kd-tree ready. starting DBSCAN.\n");

  // Rcpp objects - do not use them directly in parallel code!
  NumericMatrix d(nrow, k);
  IntegerMatrix id(nrow, k);

  NearestNeighborFinder worker(kdTree, dataPts, k, approx, d, id);
  RcppParallel::parallelFor(0, nrow, worker);

  // cleanup
  delete kdTree;
  annDeallocPts(dataPts);
  // annClose(); is now done globally in the package


  // prepare results
  List ret;
  ret["dist"] = d;
  ret["id"] = id;
  ret["k"] = k;
  ret["sort"] = true;

  // // Uncomment for testing
  // auto ret_serial = kNN_int_serial(data, k, type, bucketSize, splitRule, approx);
  // assert(ret["id"] == ret_serial["id"]);

  return ret;
}


// returns knn + dist using data and query
// [[Rcpp::export]]
List kNN_query_int(NumericMatrix data, NumericMatrix query, int k,
  int type, int bucketSize, int splitRule, double approx) {

  // FIXME: check ncol for data and query

  // copy data
  int nrow = data.nrow();
  int ncol = data.ncol();
  ANNpointArray dataPts = annAllocPts(nrow, ncol);
  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol; j++){
      (dataPts[i])[j] = data(i, j);
    }
  }

  // copy query
  int nrow_q = query.nrow();
  int ncol_q = query.ncol();
  ANNpointArray queryPts = annAllocPts(nrow_q, ncol_q);
  for(int i = 0; i < nrow_q; i++){
    for(int j = 0; j < ncol_q; j++){
      (queryPts[i])[j] = query(i, j);
    }
  }
  //Rprintf("Points copied.\n");

  // create kd-tree (1) or linear search structure (2)
  ANNpointSet* kdTree = NULL;
  if (type==1){
    kdTree = new ANNkd_tree(dataPts, nrow, ncol, bucketSize,
      (ANNsplitRule)  splitRule);
  } else{
    kdTree = new ANNbruteForce(dataPts, nrow, ncol);
  }
  //Rprintf("kd-tree ready. starting DBSCAN.\n");

  NumericMatrix d(nrow_q, k);
  IntegerMatrix id(nrow_q, k);

  // Note: does not return itself with query
  ANNdistArray dists = new ANNdist[k];
  ANNidxArray nnIdx = new ANNidx[k];

  for (int i=0; i<nrow_q; i++) {
    if (!(i % 100)) Rcpp::checkUserInterrupt();

    ANNpoint queryPt = queryPts[i];
    kdTree->annkSearch(queryPt, k, nnIdx, dists, approx);

    IntegerVector ids = IntegerVector(nnIdx, nnIdx+k);
    id(i, _) = ids + 1;

    NumericVector ndists = NumericVector(dists, dists+k);
    d(i, _) = sqrt(ndists);
  }

  // cleanup
  delete kdTree;
  delete [] dists;
  delete [] nnIdx;
  annDeallocPts(dataPts);
  annDeallocPts(queryPts);
  // annClose(); is now done globally in the package

  // prepare results (ANN returns points sorted by distance)
  List ret;
  ret["dist"] = d;
  ret["id"] = id;
  ret["k"] = k;
  ret["sort"] = true;
  return ret;
}
