#pragma once
#include <boost/serialization/access.hpp>
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "dynet/cfsm-builder.h"
#include "utils.h"
#include "mlp.h"

using namespace std;
using namespace dynet;
using namespace dynet::expr;

struct MorphLMConfig {
  bool bidirectional;
  bool use_words;
  bool use_morphology;

  unsigned word_vocab_size;
  unsigned root_vocab_size;
  unsigned affix_vocab_size;
  unsigned char_vocab_size;

  unsigned word_embedding_dim;
  unsigned root_embedding_dim; // NB: Only used for output root embeddings
  unsigned affix_embedding_dim;
  unsigned char_embedding_dim;

  unsigned model_chooser_hidden_dim;
  unsigned affix_lstm_init_hidden_dim;
  unsigned char_lstm_init_hidden_dim;

  unsigned main_lstm_dim;
  unsigned affix_lstm_dim;
  unsigned char_lstm_dim;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & bidirectional;
    ar & use_words;
    ar & use_morphology;

    ar & word_vocab_size;
    ar & root_vocab_size;
    ar & affix_vocab_size;
    ar & char_vocab_size;

    ar & word_embedding_dim;
    ar & root_embedding_dim;
    ar & affix_embedding_dim;
    ar & char_embedding_dim;

    ar & model_chooser_hidden_dim;
    ar & affix_lstm_init_hidden_dim;
    ar & char_lstm_init_hidden_dim;

    ar & main_lstm_dim;
    ar & affix_lstm_dim;
    ar & char_lstm_dim;
  }
};

class WordFillerOuter {
public:
  WordFillerOuter(Dict* word_vocab, Dict* root_vocab, Dict* affix_vocab, Dict* char_vocab) : word_vocab(word_vocab), root_vocab(root_vocab), affix_vocab(affix_vocab), char_vocab(char_vocab) {}
void FillFromChars(Sentence& sentence);
void FillFromMorph(Sentence& sentence);
void FillFromWord(Sentence& sentence);

private:
  Dict* word_vocab;
  Dict* root_vocab;
  Dict* affix_vocab;
  Dict* char_vocab;
};

class MorphLM {
public:
  MorphLM();
  ~MorphLM();
  MorphLM(Model& model, const MorphLMConfig& config);

  void NewGraph(ComputationGraph& cg);
  vector<Expression> ShowModeProbs(const Sentence& sentence, ComputationGraph& cg);
  vector<Expression> GetContexts(const vector<Expression>& inputs, ComputationGraph& cg);
  Expression BuildGraph(const Sentence& sentence, ComputationGraph& cg);
  void SetDropout(float r);

  Expression EmbedWord(const WordId word, ComputationGraph& cg);
  Expression EmbedAnalysis(const Analysis& analysis, ComputationGraph& cg);
  Expression EmbedAnalyses(const vector<Analysis>& analyses, const vector<float>& probs, ComputationGraph& cg);
  Expression EmbedCharacterSequence(const vector<WordId>& chars, ComputationGraph& cg);
  Expression EmbedInput(const Sentence& sentence, unsigned i, ComputationGraph& cg);

  Expression ComputeWordLoss(Expression context, WordId ref, ComputationGraph& cg);
  Expression ComputeAnalysisLoss(Expression context, const Analysis& ref, ComputationGraph& cg);
  Expression ComputeMorphemeLoss(Expression context, const vector<Analysis>& refs, const vector<float>& probs, ComputationGraph& cg);
  Expression ComputeCharLoss(Expression context, const vector<WordId>& ref, ComputationGraph& cg);

  Sentence Sample(unsigned max_length, ComputationGraph& cg, WordFillerOuter* wfo);
  Analysis SampleMorphAnalysis(Expression context, unsigned max_length, ComputationGraph& cg);
  vector<WordId> SampleCharSequence(Expression context, unsigned max_length, ComputationGraph& cg);

public:
  MorphLMConfig config;
//private:
  LookupParameter input_word_embeddings;
  LookupParameter input_root_embeddings;
  LookupParameter input_affix_embeddings;
  LookupParameter input_char_embeddings;

  Parameter input_char_lstm_init;
  vector<Expression> input_char_lstm_init_v;

  LSTMBuilder input_affix_lstm;
  LSTMBuilder input_char_lstm;

  Parameter main_lstm_fwd_init;
  vector<Expression> main_lstm_fwd_init_v;
  Parameter main_lstm_rev_init;
  vector<Expression> main_lstm_rev_init_v;

  LSTMBuilder main_lstm_fwd;
  LSTMBuilder main_lstm_rev;
  MLP model_chooser;

  SoftmaxBuilder* word_softmax;
  SoftmaxBuilder* root_softmax;
  SoftmaxBuilder* affix_softmax;
  SoftmaxBuilder* char_softmax;

  LookupParameter output_root_embeddings;
  LookupParameter output_affix_embeddings;
  LookupParameter output_char_embeddings;

  MLP output_affix_lstm_init;
  MLP output_char_lstm_init;

  LSTMBuilder output_affix_lstm;
  LSTMBuilder output_char_lstm;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & config;

    if (config.use_words) {
      ar & input_word_embeddings;
    }
    if (config.use_morphology) {
      ar & input_root_embeddings;
      ar & input_affix_embeddings;
    }
    ar & input_char_embeddings;

    ar & input_char_lstm_init;

    if (config.use_morphology) {
      ar & input_affix_lstm;
    }
    ar & input_char_lstm;

    ar & main_lstm_fwd_init;
    ar & main_lstm_rev_init;

    ar & main_lstm_fwd;
    ar & main_lstm_rev;
    ar & model_chooser;

    if (config.use_words) {
      ar & word_softmax;
    }
    if (config.use_morphology) {
      ar & root_softmax;
      ar & affix_softmax;
    }
    ar & char_softmax;

    if (config.use_morphology) {
      ar & output_root_embeddings;
      ar & output_affix_embeddings;
    }
    ar & output_char_embeddings;

    if (config.use_morphology) {
      ar & output_affix_lstm_init;
    }
    ar & output_char_lstm_init;

    if (config.use_morphology) {
      ar & output_affix_lstm;
    }
    ar & output_char_lstm;
  }
};

vector<Expression> MakeLSTMInitialState(Expression c, unsigned lstm_dim, unsigned lstm_layer_count);
