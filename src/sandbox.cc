#include <iostream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace boost::iostreams;

#define CHILD_STDIN_READ pipefds_input[0]
#define CHILD_STDIN_WRITE pipefds_input[1]
#define CHILD_STDOUT_READ pipefds_output[0]
#define CHILD_STDOUT_WRITE pipefds_output[1]

class Subprocess {
public:
  explicit Subprocess(const string& command);
  void write(const string& input);
  string read_line();
  string read_until_blank_line();

private:
  std::shared_ptr<boost::iostreams::stream_buffer<boost::iostreams::file_descriptor_sink>> to_child_buffer;
  std::shared_ptr<boost::iostreams::stream_buffer<boost::iostreams::file_descriptor_source>> from_child_buffer;
  std::shared_ptr<std::ostream> to_child;
  std::shared_ptr<std::istream> from_child;
};

Subprocess::Subprocess(const string& command) {
  // Create pipes for process communication
  int pipe_status;
  int pipefds_input[2];
  int pipefds_output[2];
  pipe_status = pipe(pipefds_input);
  assert (pipe_status != -1 && "Error creating pipe");
  pipe_status = pipe(pipefds_output);
  assert (pipe_status != -1 && "Error creating pipe");

  // Fork
  pid_t pid;
  pid = fork();
  if (pid == pid_t(0)) {
      // Child's IO
      dup2(CHILD_STDIN_READ, 0);
      dup2(CHILD_STDOUT_WRITE, 1);
      close(CHILD_STDIN_WRITE);
      close(CHILD_STDOUT_READ);
      // Execute external command: the format is executable followed by args
      // followed by null.  In this case, the only arg is the executable
      // itself (conventionally passed as arg0)
      execl(command.c_str(), command.c_str(), (char*) NULL);
      assert(false && "Continued after execl");
  }

  // Parent's IO
  close(CHILD_STDIN_READ);
  close(CHILD_STDOUT_WRITE);

  // IO streams for process communication
  to_child_buffer.reset(new stream_buffer<file_descriptor_sink>(CHILD_STDIN_WRITE, file_descriptor_flags::close_handle));
  from_child_buffer.reset(new stream_buffer<file_descriptor_source>(CHILD_STDOUT_READ, file_descriptor_flags::close_handle));
  to_child.reset(new ostream(to_child_buffer.get()));
  from_child.reset(new istream(from_child_buffer.get()));
}

void Subprocess::write(const string& input) {
  *to_child << input;
  to_child->flush();
}

string Subprocess::read_line() {
  string r;
  getline(*from_child, r);
  return r;
}

string Subprocess::read_until_blank_line() {
  string s;
  ostringstream ss;
  while (true) {
    getline(*from_child, s);
    if (s.size() == 0 || s == "\n") {
      break;
    }
    else {
      ss << s << endl;
    }
  }
  return ss.str();
}

int main(int argc, char** argv) {
  Subprocess sp("./omorfi.sh");
  sp.write("talossanikin\n");
  //cout << sp.read_line() << endl;
  cout << sp.read_until_blank_line() << endl;
  cout << "done." << endl;
  return 0;
}
