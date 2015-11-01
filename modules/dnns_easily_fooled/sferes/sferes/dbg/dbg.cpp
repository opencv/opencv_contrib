//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.




/*
 * File:    dbg.cpp
 * Author:  Pete Goodliffe
 * Version: 1.10
 * Created: 7 June 2001
 *
 * Purpose: C++ debugging support library
 *
 * Copyright (c) Pete Goodliffe 2001-2002 (pete@cthree.org)
 *
 * This file is modifiable/redistributable under the terms of the GNU
 * Lesser General Public License.
 *
 * You should have recieved a copy of the GNU General Public License along
 * with this program; see the file COPYING. If not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 0211-1307, USA.
 */

#ifndef DBG_ENABLED
#define DBG_ENABLED
#endif

#include "dbg.hpp"

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <new>
#include <iostream>

#include <execinfo.h>

#define MAX_RETADDR 30
void print_stack() {
  void* retaddr[MAX_RETADDR];
  char **funcnames;

  int stackdepth;

  stackdepth = backtrace(retaddr, MAX_RETADDR);
  funcnames = backtrace_symbols(retaddr, stackdepth);

  int i = 0;
  while (i < stackdepth) {
    std::cout << funcnames[i] << std::endl;
    i++;
  }
}

/**********************************************************************
 * Implementation notes
 **********************************************************************
 * Tested and found to work ok under
 *  - gcc 2.96
 *  - gcc 3.0
 *  - gcc 3.1
 *  - gcc 3.2
 *  - bcc32 5.5.1
 *  - MSVC 6.0
 *
 * MSVC v6.0
 *  - This platform makes me cry.
 *  - There are NUMEROUS hacks around it's deficient behaviour, just
 *    look for conditional complation based around _MSC_VER
 *  - The <ctime> header doesn't put all the definitions into the std
 *    namespace.
 *  - This means that we have to sacrifice our good namespace-based code
 *    for something more disgusting and primitve.
 *  - Where this has happened, and where in the future I'd really like to
 *    put the "std" namespace back in, I have instead used a STDCLK macro.
 *    See the implementation comment about this below for more grief.
 *  - A documented hack has been made in the dbg.h header file, of slightly
 *    less ghastly proportions. See dbgclock_t there.
 *  - Additionally, the dbg::array_size template utility could be (and was)
 *    more  elegantly be written:
 *         template <class T, int size>
 *         inline unsigned int array_size(T (&array)[size])
 *         {
 *             return size;
 *         }
 *    Of course, MSVC doesn't like that. Sigh. The version in dbg.h also
 *    works, its just not quite so nice.
 *  - The map implentation of MSVC doesn't provide data_type, so I have to
 *    hack around that.
 *  - The compiler doesn't like the dbg_ostream calling it's parent
 *    constructor by the name "ostream", it doesn't recognise the typedef.
 *    Ugh.
 *
 * Other thoughts:
 *  - Break out to debugger facility?
 *  - Only works for ostreams, not all basic_ostreams
 *  - Post-conditions are a bit limited, this is more of a C++
 *    language limitation, really.
 *********************************************************************/

/******************************************************************************
 * Tedious compiler-specific issues
 *****************************************************************************/

// Work around MSVC 6.0
#ifdef _MSC_VER
#define STDCLK
#pragma warning(disable:4786)
#else
// In an ideal world, the following line would be
//     namespace STDCLK = std;
// However, gcc 2.96 doesn't seem to cope well with namespace aliases.
// Sigh.
#define STDCLK std
#endif

// Quieten tedius build warnings on Borland C++ compiler
#ifdef __BCPLUSPLUS__
#pragma warn -8066
#pragma warn -8071
#pragma warn -8070
#endif

/******************************************************************************
 * General dbg library private declarations
 *****************************************************************************/

namespace {
  /**************************************************************************
   * Constants
   *************************************************************************/

  const char *LEVEL_NAMES[] = {
    "info",
    "warning",
    "error",
    "fatal",
    "tracing",
    "debug",
    "none",
    "all"
  };
  const char *BEHAVIOUR_NAMES[] = {
    "assertions_abort",
    "assertions_throw",
    "assertions_continue"
  };
  enum constraint_type {
    why_assertion,
    why_sentinel,
    why_unimplemented,
    why_check_ptr
  };

  const char         *TRACE_IN         = "->";
  const char         *TRACE_OUT        = "<-";
  const char         *INDENT           = "  ";
  const char         *PREFIX           = "*** ";
  const char         *TRUE_STRING      = "true";
  const char         *FALSE_STRING     = "false";
  const unsigned int  ALL_SOURCES_MASK = 0xff;
  const unsigned int  NUM_DBG_LEVELS   = dbg::all-1;

  /**************************************************************************
   * Internal types
   *************************************************************************/

  /**
   * Period information about a particular @ref source_pos. Used if
   * assertion periods are enabled.
   *
   * @internal
   */
  struct period_data {
    size_t          no_triggers;
    STDCLK::clock_t triggered_at;

    period_data();
  };

  /**
   * Functor to provide comparison of @ref dbg::source_pos structs.
   *
   * @internal
   */
  struct lt_sp {
    bool operator()(const dbg::source_pos &a, const dbg::source_pos &b)
    const {
      if (a.file == b.file) {
        if (a.func == b.func) {
          return a.line < b.line;
        } else {
          return a.func < b.func;
        }
      } else {
        return a.file < b.file;
      }
    }
  };

  /**
   * A handy std::streambuf that sends its input to multiple output streams.
   *
   * This is the cornerstone of the dbg output stream magic.
   *
   * The multiple streams are recorded in a std::vector. The vector is
   * external to this class, and maintained by the @ref dbg_ostream which
   * uses this streambuf.
   *
   * @internal
   */
  class dbg_streambuf : public std::streambuf {
   public:

    dbg_streambuf(std::vector<std::ostream*> &ostreams, int bsize = 0);
    ~dbg_streambuf();

    int pubsync() {
      return sync();
    }

   protected:

    int overflow(int);
    int sync();

   private:

    void put_buffer(void);
    void put_char(int);

    std::vector<std::ostream *> &ostreams;
  };

  /**
   * A handy streambuf that swallows its output and doesn't burp.
   *
   * This is used for disabled diagnostic output streams.
   *
   * There is a single stream object created using this streambuf,
   * null_ostream.
   *
   * @internal
   */
  class null_streambuf : public std::streambuf {
   public:

    null_streambuf()  {}
    ~null_streambuf() {}

   protected:

    int overflow(int) {
      return 0;
    }
    int sync()        {
      return 0;
    }
  };

  /**
   * This class provides an ostream using the @ref dbg_streambuf. It manages
   * the vector on which the @ref dbg_streambuf relies. It provides methods
   * to add/remove and clear streams.
   *
   * There are some minor hacks to support MSVC which cause small headaches.
   *
   * @internal
   */
  class dbg_ostream : public std::ostream {
   public:

#ifndef _MSC_VER
    dbg_ostream() : std::ostream(&dbg_buf), dbg_buf(streams) {}
    dbg_ostream(const dbg_ostream &rhs)
      : std::ostream(&dbg_buf), streams(rhs.streams),
        dbg_buf(streams) {}
#else
    // MSVC workaround. Sigh. It won't let us call the parent ctor as
    // "ostream" - it doesn't like the use of a typedef. On the other
    // hand gcc 2.96 doesn't provide basic_ostream, so I can't call the
    // base basic_ostream<> class there.
    dbg_ostream()
      : std::basic_ostream<char>(&dbg_buf), dbg_buf(streams) {}
    dbg_ostream(const dbg_ostream &rhs)
      : std::basic_ostream<char>(&dbg_buf), streams(rhs.streams),
        dbg_buf(streams) {}
#endif
    ~dbg_ostream() {
      dbg_buf.pubsync();
    }

    void add(std::ostream &o);
    void remove(std::ostream &o);
    void clear();

   private:

    dbg_ostream &operator=(const dbg_ostream&);

    typedef std::vector<std::ostream*> stream_vec_type;

    stream_vec_type streams;
    dbg_streambuf   dbg_buf;
  };

  /**
   * The source_info class holds all the information associated with a
   * debugging source. That is, the level enable settings and the
   * set of output streams for each level.
   *
   * source_info objects are held in the @ref source_map. The default
   * constructor magically takes settings from the @ref dbg::default_source.
   * This actually makes the constructor code a little tortuous, but it's
   * worth it.
   *
   * @internal
   */
  class source_info {
   public:

    /**
     * The source_info can either be constructed as a copy of the
     * default_source or not. The only time we choose not to is
     * when constructing default_source!
     */
    enum ConstructionStyle {
      ConstructTheDefaultSource    = 0,
      ConstructCopyOfDefaultSource = 1
    };

    source_info(ConstructionStyle cs = ConstructCopyOfDefaultSource);
    source_info(const source_info &rhs);
    ~source_info();

    /**
     * Enable or disable the given level, depending on the value
     * of the boolean.
     */
    void enable(dbg::level lvl, bool enable);

    /**
     * Returns whether or not the level is enabled.
     */
    bool enabled(dbg::level lvl) const {
      return (levels & dbg_source_mask(lvl)) != 0;
    }

    /**
     * Add an ostream to catch output at the specified level.
     */
    void add_ostream(dbg::level lvl, std::ostream &o);

    /**
     * Remove an ostream from the specified level.
     */
    void remove_ostream(dbg::level lvl, std::ostream &o);

    /**
     * Clear all ostreams from the given level.
     */
    void clear_ostream(dbg::level lvl);

    /**
     * Return a stream to write to for the given level, or the
     * @ref null_ostream if the level is not enabled for this source.
     */
    std::ostream &out(dbg::level lvl);

   private:

    /**
     * Creates a unsigned int mask that is used as the second value
     * of the sources map.
     */
    static unsigned int dbg_source_mask(dbg::level lvl) {
      return (lvl != dbg::all) ? 1 << lvl : ALL_SOURCES_MASK;
    }

    unsigned int levels;

    // We do a placement new of the dbg_streams array.
    // It looks somewhat tacky, but it allows us to have a single
    // constructor, which simplifies the client interface of this class.
    // It specifically avoids tonnes of grotesque unused dbg_ostream
    // constructions, as you'd create an array, and then copy the
    // default_source elements directly over these freshly constructed
    // elements. dbg_ostream is complex enough that this matters.
    // If we didn't have a clever cloning constructor, these lines
    // would just be "dbg_ostream dbg_streams[NUM_DBG_LEVELS];" and
    // we'd suffer a whole load of redundant dbg_ostream constructions.
    /*
      typedef        dbg_ostream array_type[NUM_DBG_LEVELS];
      dbg_ostream   *dbg_streams;
      unsigned char  raw_dbg_streams[sizeof(array_type)];
    */
    struct array_type {
      // I wrap this up in an enclosing struct to make it obvious
      // how to destroy the array "in place". To be honest I couldn't
      // figure the syntax for the corresponding delete for a
      // placement new array constrution.
      dbg_ostream dbg_streams[NUM_DBG_LEVELS];
    };
    dbg_ostream   *dbg_streams;
    unsigned char  raw_dbg_streams[sizeof(array_type)];

    array_type &raw_cast() {
      return *reinterpret_cast<array_type*>(raw_dbg_streams);
    }
    const array_type &raw_cast() const {
      return *reinterpret_cast<const array_type*>(raw_dbg_streams);
    }
  };

  /**
   * This class provides a "map" type for the source_map. It's a thin veneer
   * around a std::map. The only reason for it's existance is the ctor which
   * creates the default source_info, and inserts it into the map.
   *
   * Only the member functions that are needed below have been implemented,
   * each as forwards to the std::map methods.
   *
   * @internal
   */
  class source_map_type {
   public:

    typedef std::map<std::string, source_info> map_type;
    typedef map_type::iterator                 iterator;
    typedef map_type::key_type                 key_type;
    typedef source_info                        data_type;
    source_map_type() {
      // Insert the default_source into the map
      _map.insert(
        std::make_pair(dbg::default_source,
                       source_info(source_info::ConstructTheDefaultSource)));
      // Insert the unnamed source into the map too
      _map.insert(
        std::make_pair(dbg::dbg_source(""),
                       source_info(source_info::ConstructTheDefaultSource)));
    }
    iterator   begin()                  {
      return _map.begin();
    }
    iterator   end()                    {
      return _map.end();
    }
    data_type &operator[](key_type key) {
      return _map[key];
    }

   private:

    map_type _map;
  };

  typedef std::map<dbg::source_pos, period_data, lt_sp> period_map_type;

  /**************************************************************************
   * Internal variables
   *************************************************************************/

  // The stream to write to when no output is required.
  std::ostream null_ostream(new null_streambuf());

  dbg::assertion_behaviour behaviour[dbg::all+1] = {
    dbg::assertions_abort,
    dbg::assertions_abort,
    dbg::assertions_abort,
    dbg::assertions_abort,
    dbg::assertions_abort,
    dbg::assertions_abort,
    dbg::assertions_abort,
    dbg::assertions_abort
  };

  unsigned int    indent_depth  = 0;
  std::string     indent_prefix = PREFIX;
  bool            level_prefix  = false;
  bool            time_prefix   = false;
  STDCLK::clock_t period        = 0;
  source_map_type source_map;
  period_map_type period_map;

  /**************************************************************************
   * Function declarations
   *************************************************************************/

  /**
   * Prints a source_pos to the given ostream.
   */
  void print_pos(std::ostream &out, const dbg::source_pos &where);

  /**
   * Prints a source_pos to the given ostream in short format
   * (suitable for trace).
   */
  void print_pos_short(std::ostream &out, const dbg::source_pos &where);

  /**
   * Prints period information to the given ostream, if a period has been
   * enabled.
   */
  void print_period_info(std::ostream &out, const dbg::source_pos &where);

  /**
   * Does whatever the assertion_behaviour is set to. If an assertion
   * is triggered, then this will be called.
   */
  void do_assertion_behaviour(dbg::level lvl, constraint_type why,
                              const dbg::source_pos &pos);

  /**
   * Produces a level prefix for the specified level to the
   * given ostream.
   */
  void do_prefix(dbg::level lvl, std::ostream &s);

  /**
   * Used by period_allows below.
   */
  bool period_allows_impl(const dbg::source_pos &where);

  /**
   * Returns whether the period allows the constraint at the specified
   * @ref dbg::source_pos to trigger. This presumes that the assertion
   * at this position has shown to be broken already.
   *
   * This is a small inline function to make code that uses the period
   * implementation easier to read.
   */
  inline bool period_allows(const dbg::source_pos &where) {
    return !period || period_allows_impl(where);
  }

  /**
   * Given a dbg_source (which could be a zero pointer, or a source string)
   * and the source_pos (which could contain a DBG_SOURCE defintion, or
   * zero), work out what the dbg_source name to use is.
   */
  void determine_source(dbg::dbg_source &src, const dbg::source_pos &here);
}


/******************************************************************************
 * Miscellaneous public bobbins
 *****************************************************************************/

dbg::dbg_source dbg::default_source = "dbg::private::default_source";


/******************************************************************************
 * Enable/disable dbg facilities
 *****************************************************************************/
void dbg::init() {
  // debug
  enable_level_prefix(true);
  enable_all(dbg::all, true);
  set_prefix("|");

}

void dbg::enable(dbg::level lvl, bool enabled) {
  out(debug) << prefix(debug) << "dbg::enable(" << LEVEL_NAMES[lvl]
             << "," << (enabled ? TRUE_STRING : FALSE_STRING) << ")\n";

  source_map[""].enable(lvl, enabled);
}


void dbg::enable(dbg::level lvl, dbg::dbg_source src, bool enabled) {
  out(debug) << prefix(debug) << "dbg::enable(" << LEVEL_NAMES[lvl]
             << ",\"" << src << "\","
             << (enabled ? TRUE_STRING : FALSE_STRING) << ")\n";

  source_map[src].enable(lvl, enabled);
}


void dbg::enable_all(dbg::level lvl, bool enabled) {
  out(debug) << prefix(debug) << "dbg::enable_all("
             << LEVEL_NAMES[lvl] << ","
             << (enabled ? TRUE_STRING : FALSE_STRING) << ")\n";

  source_map_type::iterator i = source_map.begin();
  for ( ; i != source_map.end(); ++i) {
    (i->second).enable(lvl, enabled);
  }
}


/******************************************************************************
 * Logging
 *****************************************************************************/

std::ostream &dbg::out(dbg::level lvl, dbg::dbg_source src) {
  if (!src)
    return (source_map[""].out(lvl)<<prefix(lvl)<<END_COLOR);
  else
    return (source_map[src].out(lvl)<<prefix(lvl)<<" ["<<src<<"] "<<END_COLOR);
}


void dbg::attach_ostream(dbg::level lvl, std::ostream &o) {
  out(debug) << prefix(debug) << "dbg::attach_ostream("
             << LEVEL_NAMES[lvl] << ",ostream)\n";

  source_map[""].add_ostream(lvl, o);
}


void dbg::attach_ostream(dbg::level lvl, dbg::dbg_source src, std::ostream &o) {
  out(debug) << prefix(debug) << "dbg::attach_ostream("
             << LEVEL_NAMES[lvl]
             << ", \"" << src
             << "\" ,ostream)\n";

  source_map[src].add_ostream(lvl, o);
}


void dbg::detach_ostream(dbg::level lvl, std::ostream &o) {
  out(debug) << prefix(debug) << "dbg::detach_ostream("
             << LEVEL_NAMES[lvl] << ")\n";

  source_map[""].remove_ostream(lvl, o);
}


void dbg::detach_ostream(dbg::level lvl, dbg::dbg_source src, std::ostream &o) {
  out(debug) << prefix(debug) << "dbg::detach_ostream("
             << LEVEL_NAMES[lvl]
             << ", \"" << src
             << "\" ,ostream)\n";

  source_map[src].remove_ostream(lvl, o);
}


void dbg::detach_all_ostreams(dbg::level lvl) {
  out(debug) << prefix(debug) << "dbg::detach_all_ostreams("
             << LEVEL_NAMES[lvl]
             << ")\n";

  source_map[""].clear_ostream(lvl);
}


void dbg::detach_all_ostreams(dbg::level lvl, dbg::dbg_source src) {
  out(debug) << prefix(debug) << "dbg::detach_all_ostreams("
             << LEVEL_NAMES[lvl]
             << ", \"" << src << "\")\n";

  source_map[src].clear_ostream(lvl);
}


/******************************************************************************
 * Output formatting
 *****************************************************************************/

void dbg::set_prefix(const char *pfx) {
  out(debug) << prefix(debug) << "dbg::set_prefix(" << pfx << ")\n";

  indent_prefix = pfx;
}


void dbg::enable_level_prefix(bool enabled) {
  out(debug) << prefix(debug) << "dbg::enable_level_prefix("
             << (enabled ? TRUE_STRING : FALSE_STRING) << ")\n";

  level_prefix = enabled;
}


void dbg::enable_time_prefix(bool enabled) {
  out(debug) << prefix(debug) << "dbg::enable_time_prefix("
             << (enabled ? TRUE_STRING : FALSE_STRING) << ")\n";

  time_prefix = enabled;
}


std::ostream &dbg::operator<<(std::ostream &s, const prefix &p) {
  s << indent_prefix.c_str();
  do_prefix(p.l, s);
  return s;
}


std::ostream &dbg::operator<<(std::ostream &s, const indent &i) {
  s << indent_prefix.c_str();
  //do_prefix(i.l, s);
  for (unsigned int n = 0; n < indent_depth; n++) s << INDENT;
  return s;
}


std::ostream &dbg::operator<<(std::ostream &s, const source_pos &pos) {
  print_pos(s, pos);
  return s;
}


/******************************************************************************
 * Behaviour
 *****************************************************************************/

void dbg::set_assertion_behaviour(level lvl, dbg::assertion_behaviour b) {
  out(debug) << prefix(debug) << "dbg::set_assertion_behaviour("
             << LEVEL_NAMES[lvl] << "," << BEHAVIOUR_NAMES[b] << ")\n";

  if (lvl < dbg::all) {
    behaviour[lvl] = b;
  } else {
    for (int n = 0; n < dbg::all; n++) {
      behaviour[n] = b;
    }
  }
}


void dbg::set_assertion_period(dbgclock_t p) {
  out(debug) << prefix(debug) << "dbg::set_assertion_period("
             << p << ")\n";

  if (!p && period) {
    period_map.clear();
  }

  period = p;

  if (p && STDCLK::clock() == -1) {
    period = p;
    out(debug) << prefix(debug)
               << "*** WARNING ***\n"
               << "Platform does not support std::clock, and so\n"
               << "dbg::set_assertion_period is not supported.\n";
  }
}


/******************************************************************************
 * Assertion
 *****************************************************************************/

void dbg::assertion(dbg::level lvl, dbg::dbg_source src,
                    const assert_info &info) {
  determine_source(src, info);

  if (source_map[src].enabled(lvl) && !info.asserted && period_allows(info)) {
    std::ostream &o = out(lvl, src);

    o << indent(lvl) << "assertion \"" << info.text << "\" failed ";
    if (strcmp(src, "")) {
      o << "for \"" << src << "\" ";
    }
    o << "at ";
    print_pos(o, info);
    print_period_info(o, info);
    o << "\n";

    do_assertion_behaviour(lvl, why_assertion, info);
  }
}


/******************************************************************************
 * Sentinel
 *****************************************************************************/

void dbg::sentinel(dbg::level lvl, dbg::dbg_source src, const source_pos &here) {
  determine_source(src, here);

  if (source_map[src].enabled(lvl) && period_allows(here)) {
    std::ostream &o = out(lvl, src);
    o << indent(lvl) << "sentinel reached at ";
    print_pos(o, here);
    print_period_info(o, here);
    o << "\n";

    do_assertion_behaviour(lvl, why_sentinel, here);
  }
}


/******************************************************************************
 * Unimplemented
 *****************************************************************************/

void dbg::unimplemented(dbg::level lvl, dbg::dbg_source src,
                        const source_pos &here) {
  determine_source(src, here);

  if (source_map[src].enabled(lvl) && period_allows(here)) {
    std::ostream &o = out(lvl, src);
    o << indent(lvl) << "behaviour not yet implemented at ";
    print_pos(o, here);
    print_period_info(o, here);
    o << "\n";

    do_assertion_behaviour(lvl, why_unimplemented, here);
  }
}


/******************************************************************************
 * Pointer checking
 *****************************************************************************/

void dbg::check_ptr(dbg::level lvl, dbg::dbg_source src,
                    void *p, const source_pos &here) {
  determine_source(src, here);

  if (source_map[src].enabled(lvl) && p == 0 && period_allows(here)) {
    std::ostream &o = out(lvl, src);
    o << indent(lvl) << "pointer is zero at ";
    print_pos(o, here);
    print_period_info(o, here);
    o << "\n";

    do_assertion_behaviour(lvl, why_check_ptr, here);
  }
}


/******************************************************************************
 * Bounds checking
 *****************************************************************************/

void dbg::check_bounds(dbg::level lvl, dbg::dbg_source src,
                       int index, int bound, const source_pos &here) {
  determine_source(src, here);

  if (source_map[src].enabled(lvl)
      && index >= 0 && index >= bound
      && period_allows(here)) {
    std::ostream &o = out(lvl, src);
    o << indent(lvl) << "index " << index << " is out of bounds ("
      << bound << ") at ";
    print_pos(o, here);
    print_period_info(o, here);
    o << "\n";

    do_assertion_behaviour(lvl, why_check_ptr, here);
  }
}


/******************************************************************************
 * Tracing
 *****************************************************************************/

dbg::trace::trace(func_name_t name)
  : m_src(0), m_name(name), m_pos(DBG_HERE), m_triggered(false) {
  determine_source(m_src, m_pos);

  if (source_map[m_src].enabled(dbg::tracing)) {
    trace_begin();
  }
}


dbg::trace::trace(dbg_source src, func_name_t name)
  : m_src(src), m_name(name), m_pos(DBG_HERE), m_triggered(false) {
  determine_source(m_src, m_pos);

  if (source_map[m_src].enabled(dbg::tracing)) {
    trace_begin();
  }
}


dbg::trace::trace(const source_pos &where)
  : m_src(0), m_name(0), m_pos(where), m_triggered(false) {
  determine_source(m_src, m_pos);

  if (source_map[m_src].enabled(dbg::tracing)) {
    trace_begin();
  }
}


dbg::trace::trace(dbg_source src, const source_pos &where)
  : m_src(src), m_name(0), m_pos(where), m_triggered(false) {
  determine_source(m_src, m_pos);

  if (source_map[src].enabled(dbg::tracing)) {
    trace_begin();
  }
}


dbg::trace::~trace() {
  if (m_triggered) {
    trace_end();
  }
}


void dbg::trace::trace_begin() {
  std::ostream &o = out(dbg::tracing, m_src);
  o << indent(tracing);
  indent_depth++;
  o << TRACE_IN;
  if (m_name) {
    o << m_name;
  } else {
    print_pos_short(o, m_pos);
  }
  //  if (m_src && strcmp(m_src, ""))
  //  {
  //    o << " (for \"" << m_src << "\")";
  //  }
  o << std::endl;

  m_triggered = true;
}


void dbg::trace::trace_end() {
  std::ostream &o = out(dbg::tracing, m_src);
  indent_depth--;
  o << indent(tracing);
  o << TRACE_OUT;
  if (m_name) {
    o << m_name;
  } else {
    print_pos_short(o, m_pos);
  }
  // if (m_src && strcmp(m_src, ""))
  //  {
  //    o << " (for \"" << m_src << "\")";
  //  }
  o << std::endl;
}


/******************************************************************************
 * Internal implementation
 *****************************************************************************/

namespace {
  /**************************************************************************
   * dbg_streambuf
   *************************************************************************/

  dbg_streambuf::dbg_streambuf(std::vector<std::ostream*> &o, int bsize)
    : ostreams(o) {
    if (bsize) {
      char *ptr = new char[bsize];
      setp(ptr, ptr + bsize);
    } else {
      setp(0, 0);
    }
    setg(0, 0, 0);
  }

  dbg_streambuf::~dbg_streambuf() {
    sync();
    delete [] pbase();
  }

  int dbg_streambuf::overflow(int c) {
    put_buffer();
    if (c != EOF) {
      if (pbase() == epptr()) {
        put_char(c);
      } else {
        sputc(c);
      }
    }
    return 0;
  }

  int dbg_streambuf::sync() {
    put_buffer();
    return 0;
  }

  void dbg_streambuf::put_buffer(void) {
    if (pbase() != pptr()) {
      std::vector<std::ostream *>::iterator i = ostreams.begin();
      while (i != ostreams.end()) {
        (*i)->write(pbase(), pptr() - pbase());
        ++i;
      }
      setp(pbase(), epptr());
    }
  }

  void dbg_streambuf::put_char(int c) {
    std::vector<std::ostream *>::iterator i = ostreams.begin();
    while (i != ostreams.end()) {
      (**i) << static_cast<char>(c);
      ++i;
    }
  }


  /**************************************************************************
   * dbg_ostream
   *************************************************************************/

  void dbg_ostream::add(std::ostream &o) {
    if (std::find(streams.begin(), streams.end(), &o) == streams.end()) {
      streams.push_back(&o);
    }
  }

  void dbg_ostream::remove(std::ostream &o) {
    stream_vec_type::iterator i
      = std::find(streams.begin(), streams.end(), &o);
    if (i != streams.end()) {
      streams.erase(i);
    }
  }

  void dbg_ostream::clear() {
    streams.clear();
  }


  /**************************************************************************
   * source_info
   *************************************************************************/

  source_info::source_info(ConstructionStyle cs)
    : levels(cs ? source_map[dbg::default_source].levels : 0),
      dbg_streams(raw_cast().dbg_streams) {
    if (cs) {
      new (raw_dbg_streams)
      array_type(source_map[dbg::default_source].raw_cast());
    } else {
      new (raw_dbg_streams) array_type;
      // add cerr to the error and fatal levels.
      add_ostream(dbg::error, std::cerr);
      add_ostream(dbg::fatal, std::cerr);
    }
  }

  source_info::source_info(const source_info &rhs)
    : levels(rhs.levels), dbg_streams(raw_cast().dbg_streams) {
    new (raw_dbg_streams) array_type(rhs.raw_cast());
  }

  source_info::~source_info() {
    raw_cast().~array_type();
  }

  void source_info::enable(dbg::level lvl, bool status) {
    levels &= ~dbg_source_mask(lvl);
    if (status) {
      levels |= dbg_source_mask(lvl);
    }
  }

  void source_info::add_ostream(dbg::level lvl, std::ostream &o) {
    if (lvl == dbg::all) {
      for (unsigned int n = 0; n < NUM_DBG_LEVELS; ++n) {
        dbg_streams[n].add(o);
      }
    } else {
      dbg_streams[lvl].add(o);
    }
  }

  void source_info::remove_ostream(dbg::level lvl, std::ostream &o) {
    if (lvl == dbg::all) {
      for (unsigned int n = 0; n < NUM_DBG_LEVELS; ++n) {
        dbg_streams[n].remove(o);
      }
    } else {
      dbg_streams[lvl].remove(o);
    }
  }

  void source_info::clear_ostream(dbg::level lvl) {
    if (lvl == dbg::all) {
      for (unsigned int n = 0; n < NUM_DBG_LEVELS; ++n) {
        dbg_streams[n].clear();
      }
    } else {
      dbg_streams[lvl].clear();
    }
  }

  std::ostream &source_info::out(dbg::level lvl) {
    if (lvl == dbg::none || !enabled(lvl)) {
      return null_ostream;
    } else {
      return dbg_streams[lvl];
    }
  }


  /**************************************************************************
   * period_data
   *************************************************************************/

  period_data::period_data()
    : no_triggers(0), triggered_at(STDCLK::clock() - period*2) {
  }


  /**************************************************************************
   * Functions
   *************************************************************************/

  void print_pos(std::ostream &out, const dbg::source_pos &where) {
    if (where.file) {
      if (where.func) {
        out << "function: " << where.func << ", ";
      }
      out << "line: " << where.line << ", file: "    << where.file;
    }
  }

  void print_pos_short(std::ostream &out, const dbg::source_pos &where) {
    if (where.file) {
      if (where.func) {
        out << where.func << " (" << where.line
            << " in " << where.file << ")";
      } else {
        out << "function at (" << where.line
            << " in "    << where.file << ")";
      }
    }
  }

  void print_period_info(std::ostream &out, const dbg::source_pos &where) {
    if (period) {
      size_t no_triggers = period_map[where].no_triggers;
      out << " (triggered " << no_triggers << " time";
      if (no_triggers > 1) {
        out << "s)";
      } else {
        out << ")";
      }
    }
  }

  void do_assertion_behaviour(dbg::level lvl, constraint_type why,
                              const dbg::source_pos &pos) {
    switch (lvl != dbg::fatal ? behaviour[lvl] : dbg::assertions_abort) {
    case dbg::assertions_abort: {
      abort();
      break;
    }
    case dbg::assertions_throw: {
      switch (why) {
      default:
      case why_assertion: {
        throw dbg::assertion_exception(pos);
        break;
      }
      case why_sentinel: {
        throw dbg::sentinel_exception(pos);
        break;
      }
      case why_unimplemented: {
        throw dbg::unimplemented_exception(pos);
        break;
      }
      case why_check_ptr: {
        throw dbg::check_ptr_exception(pos);
        break;
      }
      }
      break;
    }
    case dbg::assertions_continue:
    default: {
      break;
    }
    }
  }

  void do_prefix(dbg::level lvl, std::ostream &s) {

    if (level_prefix) {
      switch (lvl) {
      case dbg::info:    {
        s << COL_GREEN<< "info: "<<END_COLOR;
        break;
      }
      case dbg::warning: {
        s << "warning: ";
        break;
      }
      case dbg::error: {
        s << COL_RED;
        if (time_prefix) {
          STDCLK::time_t t = STDCLK::time(0);
          if (t != -1) {
            s << std::string(STDCLK::ctime(&t), 24) << ": ";
          }
        }
        s << "error:";
        //	      print_stack();
        break;
      }
      case dbg::fatal:   {
        s << "  fatal: ";
        break;
      }
      case dbg::tracing: {
        s << COL_CYAN << "trace: " << END_COLOR;
        break;
      }
      case dbg::debug:   {
        s << "debug: ";
        break;
      }
      case dbg::none:    {
        break;
      }
      case dbg::all:     {
        s << "all: ";
        break;
      }
      }
    }
  }

  bool period_allows_impl(const dbg::source_pos &where) {
    period_data &data = period_map[where];
    data.no_triggers++;
    if (data.triggered_at < STDCLK::clock() - period) {
      data.triggered_at = STDCLK::clock();
      return true;
    } else {
      return false;
    }
  }

  void determine_source(dbg::dbg_source &src, const dbg::source_pos &here) {
    if (!src) src = "";
    if (!strcmp(src,"") && here.src) {
      src = here.src;
    }
  }
}
