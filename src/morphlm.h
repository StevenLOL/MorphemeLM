#pragma once
#include <boost/serialization/access.hpp>
#include "cnn/expr.h"
#include "utils.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class MorphLM {
public:
  MorphLM();
  MorphLM(const unsigned hidden_dim);
  Expression BuildGraph(const Sentence& sentence, ComputationGraph& cg);
  void SetDropout(float r);

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
  }
};
