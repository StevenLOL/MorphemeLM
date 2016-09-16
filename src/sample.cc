#include "cnn/cnn.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "utils.h"
#include "morphlm.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {  
  cnn::initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("model", po::value<string>()->required(), "model file, as output by train")
  ("max_length", po::value<unsigned>()->default_value(100), "Maximum length of output sentences");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  string model_filename = vm["model"].as<string>();
  unsigned max_length = vm["max_length"].as<unsigned>();

  Model cnn_model;
  MorphLM lm;
  Dict word_vocab, root_vocab, affix_vocab, char_vocab;
  Deserialize(model_filename, word_vocab, root_vocab, affix_vocab, char_vocab, lm, cnn_model);
  assert (word_vocab.is_frozen());
  assert (root_vocab.is_frozen());
  assert (affix_vocab.is_frozen());
  assert (char_vocab.is_frozen());

  while (true) {
    ComputationGraph cg;
    Sentence sample = lm.Sample(max_length, cg);
    vector<string> words;
    for (unsigned i = 0; i < sample.size(); ++i) {
      if (sample.chars[i].size() > 0) {
        for (unsigned j = 0; j < sample.chars[i].size(); ++j) {
          cout << char_vocab.convert(sample.chars[i][j]);
        }
      }
      else {
        cout << word_vocab.convert(sample.words[i]);
      }
      cout << " ";
    } 
    cout << boost::algorithm::join(words, " ") << endl;
    cout.flush();
  }

  return 0;
}
