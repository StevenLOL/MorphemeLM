#include <fstream>
#include "io.h"

vector<Sentence> ReadMorphText(const string& filename, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab) {
  vector<Sentence> corpus;
  Sentence current;

  ifstream f(filename);
  for (string line; getline(f, line);) {
    line = strip(line);
    if (line.length() == 0) {
      current.words.push_back(word_vocab.Convert("</s>"));
      Analysis eos_analysis = {root_vocab.Convert("</s>"), vector<WordId>()};
      current.analyses.push_back(vector<Analysis>(1, eos_analysis));
      current.analysis_probs.push_back(vector<float>(1, 1.0f));
      current.chars.push_back(vector<WordId>(1, char_vocab.Convert("</s>")));

      assert (current.words.size() == current.analyses.size());
      assert (current.words.size() == current.analysis_probs.size());
      assert (current.words.size() == current.chars.size());

      if (current.size() > 0) {
        corpus.push_back(current);
        current = Sentence();
      }
      continue;
    }

    vector<string> pieces = tokenize(line, "\t");
    assert (pieces.size() % 2 == 1);
    assert (pieces.size() >= 3);

    string& word = pieces[0];
    current.words.push_back(word_vocab.Convert(word));
    
    current.analyses.push_back(vector<Analysis>());
    current.analysis_probs.push_back(vector<float>());

    for (unsigned i = 1; i < pieces.size(); i += 2) {
      vector<string> morphemes = tokenize(pieces[i], "+");
      assert (morphemes.size() >= 1);

      string root = morphemes[0];
      morphemes.erase(morphemes.begin());

      if (root == "*UNKNOWN*") {
        root = "UNK";
      }

      Analysis analysis;
      analysis.root = root_vocab.Convert(root);
      for (string& morpheme : morphemes) {
        analysis.affixes.push_back(affix_vocab.Convert(morpheme));
      }
      analysis.affixes.push_back(affix_vocab.Convert("</w>"));
      current.analyses.back().push_back(analysis);

      float prob = atof(pieces[i + 1].c_str());
      current.analysis_probs.back().push_back(prob);
    }

    current.chars.push_back(vector<WordId>());
    unsigned i = 0;
    while (i < word.length()) {
      unsigned len = UTF8Len(word[i]);
      string c = word.substr(i, len);
      current.chars.back().push_back(char_vocab.Convert(c));
      i += len;
    }
    current.chars.back().push_back(char_vocab.Convert("</w>"));
    assert (i == word.length());
  }
  f.close();

  current.words.push_back(word_vocab.Convert("</s>"));
  Analysis eos_analysis = {root_vocab.Convert("</s>"), vector<WordId>()};
  current.analyses.push_back(vector<Analysis>(1, eos_analysis));
  current.analysis_probs.push_back(vector<float>(1, 1.0f));
  current.chars.push_back(vector<WordId>(1, char_vocab.Convert("</s>")));

  assert (current.words.size() == current.analyses.size());
  assert (current.words.size() == current.analysis_probs.size());
  assert (current.words.size() == current.chars.size());

  if (current.size() != 0) {
    corpus.push_back(current);
  }

  return corpus;
}

void Serialize(const Dict& word_vocab, const Dict& root_vocab, const Dict& affix_vocab, const Dict& char_vocab, const MorphLM& lm, Model& cnn_model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & cnn_model;
  oa & word_vocab;
  oa & root_vocab;
  oa & affix_vocab;
  oa & char_vocab;
  oa & lm;
}

void Deserialize(const string& filename, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, MorphLM& lm, Model& cnn_model) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & cnn_model;
  ia & word_vocab;
  ia & root_vocab;
  ia & affix_vocab;
  ia & char_vocab;
  ia & lm;
  f.close();

  assert (word_vocab.is_frozen());
  assert (root_vocab.is_frozen());
  assert (affix_vocab.is_frozen());
  assert (char_vocab.is_frozen());
}

