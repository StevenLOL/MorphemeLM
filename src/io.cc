#include <fstream>
#include "io.h"

bool ReadVocab(const string& filename, Dict& vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    cerr << "Error reading vocab file " << filename << endl;
    assert (f.is_open());
    return false;
  }

  for (string line; getline(f, line);) {
    line = strip(line);
    vocab.convert(line);
  }
  return true;
}

void HandleMorphLine(const string& line, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, Sentence& out) {
  vector<string> pieces = tokenize(line, "\t");
  assert (pieces.size() % 2 == 1);
  assert (pieces.size() >= 3);

  string& word = pieces[0];
  out.words.push_back(word_vocab.convert(word));
  
  out.analyses.push_back(vector<Analysis>());
  out.analysis_probs.push_back(vector<float>());

  for (unsigned i = 1; i < pieces.size(); i += 2) {
    vector<string> morphemes = tokenize(pieces[i], "+");
    assert (morphemes.size() >= 1);

    string root = morphemes[0];
    morphemes.erase(morphemes.begin());

    if (root == "*UNKNOWN*") {
      root = "UNK";
    }

    Analysis analysis;
    analysis.root = root_vocab.convert(root);
    for (string& morpheme : morphemes) {
      analysis.affixes.push_back(affix_vocab.convert(morpheme));
    }
    analysis.affixes.push_back(affix_vocab.convert("</w>"));
    out.analyses.back().push_back(analysis);

    float prob = atof(pieces[i + 1].c_str());
    out.analysis_probs.back().push_back(prob);
  }

  out.chars.push_back(vector<WordId>());
  unsigned i = 0;
  while (i < word.length()) {
    unsigned len = UTF8Len(word[i]);
    string c = word.substr(i, len);
    out.chars.back().push_back(char_vocab.convert(c));
    i += len;
  }
  out.chars.back().push_back(char_vocab.convert("</w>"));
  assert (i == word.length());
}


void EndMorphSentence(Dict& word_vocab, Dict& root_vocab, Dict& char_vocab, Sentence& out) {
  out.words.push_back(word_vocab.convert("</s>"));

  Analysis eos_analysis = {root_vocab.convert("</s>"), vector<WordId>()};
  out.analyses.push_back(vector<Analysis>(1, eos_analysis));
  out.analysis_probs.push_back(vector<float>(1, 1.0f));

  out.chars.push_back(vector<WordId>(1, char_vocab.convert("</s>")));

  assert (out.words.size() == out.analyses.size());
  assert (out.words.size() == out.analysis_probs.size());
  assert (out.words.size() == out.chars.size());
}

bool ReadMorphSentence(istream& f, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, Sentence& out) {
  out.words.clear();
  out.analyses.clear();
  out.analysis_probs.clear();
  out.chars.clear();

  for (string line; getline(f, line);) {
    line = strip(line);
    if (line.length() == 0) {
      EndMorphSentence(word_vocab, root_vocab, char_vocab, out);
      return true;
    }
    HandleMorphLine(line, word_vocab, root_vocab, affix_vocab, char_vocab, out);
  }

  if (out.size() > 0) {
    EndMorphSentence(word_vocab, root_vocab, char_vocab, out);
    return true;
  }
  return false;
}

vector<Sentence> ReadMorphText(const string& filename, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab) {
  vector<Sentence> corpus;
  Sentence current;

  ifstream f(filename);
  unsigned line_number = 1;
  for (string line; getline(f, line); ++line_number) {
    line = strip(line);
    if (line.length() == 0) {
      current.words.push_back(word_vocab.convert("</s>"));
      Analysis eos_analysis = {root_vocab.convert("</s>"), vector<WordId>()};
      current.analyses.push_back(vector<Analysis>(1, eos_analysis));
      current.analysis_probs.push_back(vector<float>(1, 1.0f));
      current.chars.push_back(vector<WordId>(1, char_vocab.convert("</s>")));

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
    if (pieces.size() % 2 != 1) {
      cerr << "Issue in " << filename << " on line " << line_number << "." << endl;
      cerr << "Offending line: " << line << " (" << pieces.size() << " pieces)" << endl;
    }
    assert (pieces.size() % 2 == 1);
    assert (pieces.size() >= 3);

    string& word = pieces[0];
    current.words.push_back(word_vocab.convert(word));
    
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
      analysis.root = root_vocab.convert(root);
      for (string& morpheme : morphemes) {
        analysis.affixes.push_back(affix_vocab.convert(morpheme));
      }
      analysis.affixes.push_back(affix_vocab.convert("</w>"));
      current.analyses.back().push_back(analysis);

      float prob = atof(pieces[i + 1].c_str());
      current.analysis_probs.back().push_back(prob);
    }

    current.chars.push_back(vector<WordId>());
    unsigned i = 0;
    while (i < word.length()) {
      unsigned len = UTF8Len(word[i]);
      string c = word.substr(i, len);
      current.chars.back().push_back(char_vocab.convert(c));
      i += len;
    }
    current.chars.back().push_back(char_vocab.convert("</w>"));
    assert (i == word.length());
  }
  f.close();

  current.words.push_back(word_vocab.convert("</s>"));
  Analysis eos_analysis = {root_vocab.convert("</s>"), vector<WordId>()};
  current.analyses.push_back(vector<Analysis>(1, eos_analysis));
  current.analysis_probs.push_back(vector<float>(1, 1.0f));
  current.chars.push_back(vector<WordId>(1, char_vocab.convert("</s>")));

  assert (current.words.size() == current.analyses.size());
  assert (current.words.size() == current.analysis_probs.size());
  assert (current.words.size() == current.chars.size());

  if (current.size() != 0) {
    corpus.push_back(current);
  }

  return corpus;
}

void Serialize(const Dict& word_vocab, const Dict& root_vocab, const Dict& affix_vocab, const Dict& char_vocab, const MorphLM& lm, Model& dynet_model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {}
  fseek(stdout, 0, SEEK_SET);

  boost::archive::binary_oarchive oa(cout);
  oa & dynet_model;
  oa & word_vocab;
  oa & root_vocab;
  oa & affix_vocab;
  oa & char_vocab;
  oa & lm;
}

void Deserialize(const string& filename, Dict& word_vocab, Dict& root_vocab, Dict& affix_vocab, Dict& char_vocab, MorphLM& lm, Model& dynet_model) {
  ifstream f(filename);
  boost::archive::binary_iarchive ia(f);
  ia & dynet_model;
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

