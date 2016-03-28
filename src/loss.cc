#include "cnn/cnn.h"
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>

#include "io.h"
#include "utils.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model files, as output by train")
  ("perp,p", "Show model perplexity instead of negative log loss")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string model_filename = vm["model"].as<string>();
  const bool show_perp = vm.count("perp") > 0;

  Model cnn_model;
  Dict word_vocab, root_vocab, affix_vocab, char_vocab;
  MorphLM lm;
  cerr << "Loading model from " << model_filename << "...";
  Deserialize(model_filename, word_vocab, root_vocab, affix_vocab, char_vocab, lm, cnn_model);
  cerr << " Done!" << endl;

  unsigned sentence_number = 0;
  cnn::real total_loss = 0;
  unsigned total_words = 0;
  Sentence input;
  while(ReadMorphSentence(cin, word_vocab, root_vocab, affix_vocab, char_vocab, input)) {
    ComputationGraph cg;
    Expression loss_expr = lm.BuildGraph(input, cg);
    cg.incremental_forward();
    cnn::real loss = as_scalar(loss_expr.value());
    unsigned words = input.size();
    if (show_perp) {
      cout << exp(loss / words) << endl;
    }
    else {
      cout << loss << endl;
    }
    cout.flush();

    sentence_number++;
    total_loss += loss;
    total_words += words;
  }

  if (show_perp) {
    cout << "Total: " << exp(total_loss / total_words) << endl;
  }
  else {
    cout << "Total: " << total_loss << endl;
  }

  return 0;
}
