#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <vector>
#include "cnn/dict.h"
#include "morphlm.h"
#include "utils.h"

using namespace std;
using namespace cnn;

vector<Sentence> ReadMorphText(const string& filename, Dict& vocab);
void Serialize(const Dict& dict, const MorphLM& translator, Model& cnn_model);
void Deserialize(const string& filename, Dict& dict, MorphLM& lm, Model& cnn_model);
