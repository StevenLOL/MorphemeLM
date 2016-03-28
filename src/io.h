#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <vector>
#include "cnn/dict.h"
#include "morphlm.h"
#include "utils.h"

using namespace std;
using namespace cnn;

bool ReadMorphSentence(istream& f, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, Sentence& out);
vector<Sentence> ReadMorphText(const string& filename, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab);
void Serialize(const Dict& word_vocab, const Dict& root_vocab, const Dict& affix_vocab, const Dict& char_vocab, const MorphLM& translator, Model& cnn_model);
void Deserialize(const string& filename, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, MorphLM& lm, Model& cnn_model);
