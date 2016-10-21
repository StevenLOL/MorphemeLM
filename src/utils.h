#pragma once
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "dynet/dict.h"

using namespace std;
using namespace dynet;

typedef int WordId;

struct Analysis {
  WordId root;
  vector<WordId> affixes;
};

struct Sentence {
  vector<WordId> words;
  vector<vector<Analysis>> analyses;
  vector<vector<float>> analysis_probs;
  vector<vector<WordId>> chars;

  unsigned size() const;
};

unsigned int UTF8Len(unsigned char x);
unsigned int UTF8StringLen(const string& x);

vector<string> tokenize(string input, string delimiter, unsigned max_times);
vector<string> tokenize(string input, string delimiter);
vector<string> tokenize(string input, char delimiter);

string strip(const string& input);
vector<string> strip(const vector<string>& input, bool removeEmpty = false);

map<string, double> parse_feature_string(string input);

float logsumexp(const vector<float>& v);

template <typename RNG>
unsigned sample_multinomial(const vector<float>& logprobs, RNG& rng) {
  assert (logprobs.size() > 0);
  float M = logprobs[0];
  for (unsigned i = 1; i < logprobs.size(); ++i) {
    M = max(logprobs[i], M);
  }

  float sum = 0.0;
  for (unsigned i = 0; i < logprobs.size(); ++i) {
    sum += exp(logprobs[i] - M);
  }

  uniform_real_distribution<float> dist(0, 1);
  float r = sum * dist(rng);
  for (unsigned i = 0; i < logprobs.size(); ++i) {
    float p = exp(logprobs[i] - M);
    if (r < p) {
      return i;
    }
    else {
      r -= p;
    }
  }
  return logprobs.size() - 1;
}

