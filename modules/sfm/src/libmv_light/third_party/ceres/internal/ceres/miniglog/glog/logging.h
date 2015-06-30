// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: settinger@google.com (Scott Ettinger)
//         keir@google.com (Keir Mierle)
//
// Simplified Glog style logging with Android support. Supported macros in
// decreasing severity level per line:
//
//   VLOG(2), VLOG(N)
//   VLOG(1),
//   LOG(INFO), VLOG(0), LG
//   LOG(WARNING),
//   LOG(ERROR),
//   LOG(FATAL),
//
// With VLOG(n), the output is directed to one of the 5 Android log levels:
//
//   2 - Verbose
//   1 - Debug
//   0 - Info
//  -1 - Warning
//  -2 - Error
//  -3 - Fatal
//
// Any logging of level 2 and above is directed to the Verbose level. All
// Android log output is tagged with the string "native".
//
// If the symbol ANDROID is not defined, all output goes to std::cerr.
// This allows code to be built on a different system for debug.
//
// Portions of this code are taken from the GLOG package.  This code is only a
// small subset of the GLOG functionality. Notable differences from GLOG
// behavior include lack of support for displaying unprintable characters and
// lack of stack trace information upon failure of the CHECK macros.  On
// non-Android systems, log output goes to std::cerr and is not written to a
// file.
//
// CHECK macros are defined to test for conditions within code.  Any CHECK that
// fails will log the failure and terminate the application.
// e.g. CHECK_GE(3, 2) will pass while CHECK_GE(3, 4) will fail after logging
//      "Check failed 3 >= 4".
//
// The following CHECK macros are defined:
//
//   CHECK(condition)        - fails if condition is false and logs condition.
//   CHECK_NOTNULL(variable) - fails if the variable is NULL.
//
// The following binary check macros are also defined :
//
//   Macro                     Operator equivalent
//   --------------------      -------------------
//   CHECK_EQ(val1, val2)      val1 == val2
//   CHECK_NE(val1, val2)      val1 != val2
//   CHECK_GT(val1, val2)      val1 > val2
//   CHECK_GE(val1, val2)      val1 >= val2
//   CHECK_LT(val1, val2)      val1 < val2
//   CHECK_LE(val1, val2)      val1 <= val2
//
// Debug only versions of all of the check macros are also defined.  These
// macros generate no code in a release build, but avoid unused variable
// warnings / errors.
//
// To use the debug only versions, prepend a D to the normal check macros, e.g.
// DCHECK_EQ(a, b).

#ifndef CERCES_INTERNAL_MINIGLOG_GLOG_LOGGING_H_
#define CERCES_INTERNAL_MINIGLOG_GLOG_LOGGING_H_

#ifdef ANDROID
#include <android/log.h>
#endif  // ANDROID

#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

// Log severity level constants.
const int FATAL   = -3;
const int ERROR   = -2;
const int WARNING = -1;
const int INFO    =  0;

// ------------------------- Glog compatibility ------------------------------

namespace google {

typedef int LogSeverity;
const int INFO    = ::INFO;
const int WARNING = ::WARNING;
const int ERROR   = ::ERROR;
const int FATAL   = ::FATAL;

// Sink class used for integration with mock and test functions. If sinks are
// added, all log output is also sent to each sink through the send function.
// In this implementation, WaitTillSent() is called immediately after the send.
// This implementation is not thread safe.
class LogSink {
 public:
  virtual ~LogSink() {}
  virtual void send(LogSeverity severity,
                    const char* full_filename,
                    const char* base_filename,
                    int line,
                    const struct tm* tm_time,
                    const char* message,
                    size_t message_len) = 0;
  virtual void WaitTillSent() = 0;
};

// Global set of log sinks. The actual object is defined in logging.cc.
extern std::set<LogSink *> log_sinks_global;

inline void InitGoogleLogging(char *argv) {
  // Do nothing; this is ignored.
}

// Note: the Log sink functions are not thread safe.
inline void AddLogSink(LogSink *sink) {
  // TODO(settinger): Add locks for thread safety.
  log_sinks_global.insert(sink);
}
inline void RemoveLogSink(LogSink *sink) {
  log_sinks_global.erase(sink);
}

}  // namespace google

// ---------------------------- Logger Class --------------------------------

// Class created for each use of the logging macros.
// The logger acts as a stream and routes the final stream contents to the
// Android logcat output at the proper filter level.  If ANDROID is not
// defined, output is directed to std::cerr.  This class should not
// be directly instantiated in code, rather it should be invoked through the
// use of the log macros LG, LOG, or VLOG.
class MessageLogger {
 public:
  MessageLogger(const char *file, int line, const char *tag, int severity)
    : file_(file), line_(line), tag_(tag), severity_(severity) {
    // Pre-pend the stream with the file and line number.
    StripBasename(std::string(file), &filename_only_);
    stream_ << filename_only_ << ":" << line << " ";
  }

  // Output the contents of the stream to the proper channel on destruction.
  ~MessageLogger() {
    stream_ << "\n";

#ifdef ANDROID
    static const int android_log_levels[] = {
        ANDROID_LOG_FATAL,    // LOG(FATAL)
        ANDROID_LOG_ERROR,    // LOG(ERROR)
        ANDROID_LOG_WARN,     // LOG(WARNING)
        ANDROID_LOG_INFO,     // LOG(INFO), LG, VLOG(0)
        ANDROID_LOG_DEBUG,    // VLOG(1)
        ANDROID_LOG_VERBOSE,  // VLOG(2) .. VLOG(N)
    };

    // Bound the logging level.
    const int kMaxVerboseLevel = 2;
    int android_level_index = std::min(std::max(FATAL, severity_),
                                       kMaxVerboseLevel) - FATAL;
    int android_log_level = android_log_levels[android_level_index];

    // Output the log string the Android log at the appropriate level.
    __android_log_print(android_log_level, tag_.c_str(), stream_.str().c_str());

    // Indicate termination if needed.
    if (severity_ == FATAL) {
      __android_log_print(ANDROID_LOG_FATAL,
                          tag_.c_str(),
                          "terminating.\n");
    }
#else
    // If not building on Android, log all output to std::cerr.
    std::cerr << stream_.str();
#endif  // ANDROID

    LogToSinks(severity_);
    WaitForSinks();

    // Android logging at level FATAL does not terminate execution, so abort()
    // is still required to stop the program.
    if (severity_ == FATAL) {
      abort();
    }
  }

  // Return the stream associated with the logger object.
  std::stringstream &stream() { return stream_; }

 private:
  void LogToSinks(int severity) {
    time_t rawtime;
    struct tm* timeinfo;

    time (&rawtime);
    timeinfo = localtime(&rawtime);
    std::set<google::LogSink*>::iterator iter;
    // Send the log message to all sinks.
    for (iter = google::log_sinks_global.begin();
         iter != google::log_sinks_global.end(); ++iter) {
      (*iter)->send(severity, file_.c_str(), filename_only_.c_str(), line_,
                    timeinfo, stream_.str().c_str(), stream_.str().size());
    }
  }

  void WaitForSinks() {
    // TODO(settinger): Add locks for thread safety.
    std::set<google::LogSink *>::iterator iter;

    // Call WaitTillSent() for all sinks.
    for (iter = google::log_sinks_global.begin();
         iter != google::log_sinks_global.end(); ++iter) {
      (*iter)->WaitTillSent();
    }
  }

  void StripBasename(const std::string &full_path, std::string *filename) {
    // TODO(settinger): add support for OS with different path separators.
    const char kSeparator = '/';
    size_t pos = full_path.rfind(kSeparator);
    if (pos != std::string::npos) {
      *filename = full_path.substr(pos + 1, std::string::npos);
    } else {
      *filename = full_path;
    }
  }

  std::string file_;
  std::string filename_only_;
  int line_;
  std::string tag_;
  std::stringstream stream_;
  int severity_;
};

// ---------------------- Logging Macro definitions --------------------------

#define LG MessageLogger((char *)__FILE__, __LINE__, "native", \
                         INFO).stream()

#define LOG(n) MessageLogger((char *)__FILE__, __LINE__, "native", \
                             n).stream()

#define VLOG(n) MessageLogger((char *)__FILE__, __LINE__, "native", \
                              n).stream()

// Currently, VLOG is always on.
#define VLOG_IS_ON(x) true

// ---------------------------- CHECK helpers --------------------------------

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LoggerVoidify {
 public:
  LoggerVoidify() { }
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(const std::ostream &s) { }
};

// Log only if condition is met.  Otherwise evaluates to void.
#define LOG_IF(severity, condition) \
  condition ? (void) 0 : LoggerVoidify() & LOG(severity)

// Log a message and terminate.
template<class T>
void LogMessageFatal(const char *file, int line, const T &message) {
  MessageLogger((char *)__FILE__, __LINE__, "native", FATAL).stream()
      << message;
}

// ---------------------------- CHECK macros ---------------------------------

// Check for a given boolean condition.
#define CHECK(condition) LOG_IF(FATAL, condition) \
        << "Check failed: " #condition " "

#ifndef NDEBUG
// Debug only version of CHECK
#define DCHECK(condition) LOG_IF(FATAL, condition) \
        << "Check failed: " #condition " "
#else
// Optimized version - generates no code.
#define DCHECK(condition) if (false) LOG_IF(FATAL, condition) \
        << "Check failed: " #condition " "
#endif  // NDEBUG

// ------------------------- CHECK_OP macros ---------------------------------

// Generic binary operator check macro. This should not be directly invoked,
// instead use the binary comparison macros defined below.
#define CHECK_OP(val1, val2, op) LOG_IF(FATAL, (val1 op val2)) \
  << "Check failed: " #val1 " " #op " " #val2 " "

// Check_op macro definitions
#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)

#ifndef NDEBUG
// Debug only versions of CHECK_OP macros.
#define DCHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#else
// These versions generate no code in optimized mode.
#define DCHECK_EQ(val1, val2) if (false) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) if (false) CHECK_OP(val1, val2, !=)
#define DCHECK_LE(val1, val2) if (false) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) if (false) CHECK_OP(val1, val2, <)
#define DCHECK_GE(val1, val2) if (false) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) if (false) CHECK_OP(val1, val2, >)
#endif  // NDEBUG

// ---------------------------CHECK_NOTNULL macros ---------------------------

// Helpers for CHECK_NOTNULL(). Two are necessary to support both raw pointers
// and smart pointers.
template <typename T>
T& CheckNotNullCommon(const char *file, int line, const char *names, T& t) {
  if (t == NULL) {
    LogMessageFatal(file, line, std::string(names));
  }
  return t;
}

template <typename T>
T* CheckNotNull(const char *file, int line, const char *names, T* t) {
  return CheckNotNullCommon(file, line, names, t);
}

template <typename T>
T& CheckNotNull(const char *file, int line, const char *names, T& t) {
  return CheckNotNullCommon(file, line, names, t);
}

// Check that a pointer is not null.
#define CHECK_NOTNULL(val) \
  CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// Debug only version of CHECK_NOTNULL
#define DCHECK_NOTNULL(val) \
  CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))
#else
// Optimized version - generates no code.
#define DCHECK_NOTNULL(val) if (false)\
  CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))
#endif  // NDEBUG

#endif  // CERCES_INTERNAL_MINIGLOG_GLOG_LOGGING_H_
