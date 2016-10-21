#include "train.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

class Learner : public ILearner<Sentence, SufficientStats> {
public:
  Learner(Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, MorphLM& lm, Model& dynet_model) :
    word_vocab(word_vocab), root_vocab(root_vocab), affix_vocab(affix_vocab), char_vocab(char_vocab), lm(lm), dynet_model(dynet_model) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const Sentence& datum, bool learn) {
    ComputationGraph cg;
    if (learn) {
      lm.SetDropout(dropout_rate);
    }
    else {
      lm.SetDropout(0.0f);
    }
    Expression loss_expr = lm.BuildGraph(datum, cg);
    dynet::real loss = as_scalar(cg.forward(loss_expr));
    if (learn) {
      cg.backward(loss_expr);
    }
    return SufficientStats(loss, datum.size(), 1);
  }

  void SaveModel() {
    if (!quiet) {
      Serialize(word_vocab, root_vocab, affix_vocab, char_vocab, lm, dynet_model);
    }
  }

  bool quiet;
  float dropout_rate;
private:
  Dict& word_vocab;
  Dict& root_vocab;
  Dict& affix_vocab;
  Dict& char_vocab;
  MorphLM& lm;
  Model& dynet_model;
};

// This function lets us elegantly handle the user pressing ctrl-c.
// We set a global flag, which causes the training loops to clean up
// and break. In particular, this allows models to be saved to disk
// before actually exiting the program.
bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
    dynet::mp::stop_requested = true;
  }
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("train_text", po::value<string>()->required(), "Training text, morphologically analyzed")
  ("dev_text", po::value<string>()->required(), "Dev text, used for early stopping")
  ("word_vocab", po::value<string>()->required(), "Surface form vocab list of words. Anything outside this list must be generated via morphology or characters")
  ("root_vocab", po::value<string>()->required(), "Vocabulary of word stems. Anything outside this list must be generated as a character stream (or maybe as whole words, but probably not)")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("dropout_rate", po::value<float>()->default_value(0.0), "Dropout rate (should be >= 0.0 and < 1)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best")
  ("quiet,q", "Do not output model")
  ("no_words,W", "Do not use word-level information")
  ("no_morphology,M", "Do not use morpheme-level information")
  ("model", po::value<string>(), "Reload this model and continue learning");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("train_text", 1);
  positional_options.add("dev_text", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const string train_text_filename = vm["train_text"].as<string>();
  const string dev_text_filename = vm["dev_text"].as<string>();
  const string word_vocab_filename = vm["word_vocab"].as<string>();
  const string root_vocab_filename = vm["root_vocab"].as<string>();

  Dict word_vocab, root_vocab, affix_vocab, char_vocab;
  Model dynet_model;
  MorphLM* lm = nullptr;
  Trainer* trainer = nullptr;

  if (vm.count("model")) {
    lm = new MorphLM();
    string model_filename = vm["model"].as<string>();
    Deserialize(model_filename, word_vocab, root_vocab, affix_vocab, char_vocab, *lm, dynet_model);
    assert (word_vocab.is_frozen());
    assert (root_vocab.is_frozen());
    assert (affix_vocab.is_frozen());
    assert (char_vocab.is_frozen());
  }

  // TODO: Do we need start symbols at all?
  word_vocab.convert("UNK");
  word_vocab.convert("<s>");
  word_vocab.convert("</s>");
  root_vocab.convert("UNK");
  root_vocab.convert("<s>");
  root_vocab.convert("</s>");
  char_vocab.convert("UNK");
  char_vocab.convert("<s>");
  char_vocab.convert("</s>");
  char_vocab.convert("</w>");
  affix_vocab.convert("UNK");
  affix_vocab.convert("</w>");

  ReadVocab(word_vocab_filename, word_vocab);
  ReadVocab(root_vocab_filename, root_vocab);
  word_vocab.freeze();
  word_vocab.set_unk("UNK");
  root_vocab.freeze();
  root_vocab.set_unk("UNK");

  vector<Sentence> train_text = ReadMorphText(train_text_filename, word_vocab, root_vocab, affix_vocab, char_vocab); 

  if (!vm.count("model")) {
    MorphLMConfig config;
    config.use_words = (vm.count("no_words") == 0);
    config.use_morphology = (vm.count("no_morphology") == 0);
    config.word_vocab_size = word_vocab.size();
    config.root_vocab_size = root_vocab.size();
    config.affix_vocab_size = affix_vocab.size();
    config.char_vocab_size = char_vocab.size();
    config.word_embedding_dim = 128; // 3 32 128
    config.root_embedding_dim = 128; // 3 32 128
    config.affix_embedding_dim = 64; // 1 16 64
    config.char_embedding_dim = 64; // 1 16 64
    config.model_chooser_hidden_dim = 16; // 1 16 16
    config.affix_lstm_init_hidden_dim = 128; // 1 16 128
    config.char_lstm_init_hidden_dim = 64; // 1 16 64
    config.main_lstm_dim = 256; // 6 64 256
    config.affix_lstm_dim = 128; // 1 16 128
    config.char_lstm_dim = 64; // 1 16 64
    // Maybe only need 1 layer on input LSTMs
    lm = new MorphLM(dynet_model, config);

    affix_vocab.freeze();
    affix_vocab.set_unk("UNK");
    char_vocab.freeze();
    char_vocab.set_unk("UNK");
    cerr << "Dicts frozen" << endl;
  }

  vector<Sentence> dev_text = ReadMorphText(dev_text_filename, word_vocab, root_vocab, affix_vocab, char_vocab);

  cerr << "Vocabulary sizes: " << word_vocab.size() << " words, " << root_vocab.size() << " roots, " << affix_vocab.size() << " affixes, " << char_vocab.size() << " chars" << endl;
  cerr << "Total parameters: " << dynet_model.parameter_count() << endl;

  trainer = CreateTrainer(dynet_model, vm);
  Learner learner(word_vocab, root_vocab, affix_vocab, char_vocab, *lm, dynet_model);
  learner.quiet = vm.count("quiet") > 0;
  learner.dropout_rate = vm["dropout_rate"].as<float>();
  unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  unsigned report_frequency = vm["report_frequency"].as<unsigned>();
  if (num_cores > 1) {
    run_multi_process<Sentence>(num_cores, &learner, trainer, train_text, dev_text, num_iterations, dev_frequency, report_frequency);
  }
  else {
    run_single_process<Sentence>(&learner, trainer, train_text, dev_text, num_iterations, dev_frequency, report_frequency, 1);
  }

  return 0;
}
