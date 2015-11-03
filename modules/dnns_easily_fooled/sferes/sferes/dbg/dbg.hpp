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
 * File:    dbg.h
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

#ifndef DBG_DBG_H
#define DBG_DBG_H

#include <iosfwd>
#include <exception>
#include <cstring>
#include <cstdio>

#ifndef DBG_COLORS
#define DBG_COLORS
#define COL_RED "\033[1;33;41m"
#define COL_GREEN "\033[32m"
#define COL_BLACK "\033[30m"
#define COL_ORANGE "\033[33m"
#define COL_BLUE "\033[34m"
#define COL_MAGENTA "\033[35m"
#define COL_CYAN "\033[36m"
#define END_COLOR "\033[m"
#endif

#ifndef _MSC_VER
#include <ctime>
#else
// The start of a MSVC compatibility disaster area.
// See the documentation for the dbgclock_t type.
#include <time.h>
#endif

#if defined(DBG_ENABLED) && defined(NDEBUG)
//#warning DBG_ENABLED defined with NDEBUG which do you want?
#endif

/**
 * @libdoc dbg library
 *
 * The dbg library is a set of C++ utilities to facilitate modern debugging
 * idioms.
 *
 * It has been designed to support defensive programming techniques in modern
 * C++ code. It integrates well with standard library usage and has been
 * carefully designed to be easy to write, easy to read and very easy to use.
 *
 * It provides various constraint checking utilities together with an
 * integrated error logging facility. These utilities are flexible and
 * customisable. They can be enabled and disabled at runtime, and in release
 * builds, dbg library use can be compiled away to nothing.
 *
 * Rich debugging can only be implemented in large code bases from the outset,
 * it is hard to retrofit full defensive programming techniques onto existant
 * code. For this reason it is good practice to use a library like dbg when
 * you start a new project. By using dbg extensively you will find bugs
 * quicker, and prevent more insideous problems rearing their head later in
 * the project's life.
 *
 * For instructions on the dbg library's use see the @ref dbg namespace
 * documentation.
 */



/**
 * The dbg namespace holds a number of C++ debugging utilities.
 *
 * They allow you to include constraint checking in your code, and provide
 * an integrated advanced stream-based logging facility.
 *
 * The characteristics of this library are:
 *   @li Easy to use (not overly complex)
 *       (easy to write, easy to read, easy to use)
 *   @li Powerful
 *   @li Configurable
 *   @li No run time overhead when "compiled out"
 *   @li Minimises use of the proprocessor
 *   @li Can throw exceptions if required
 *   @li Can separate different "sources" of diagnostic output (these
 *       sources are differentated by name)
 *   @li Designed to be a "standard" library
 *       (integrates with the style of the C++ standard library and works
 *       well with it)
 *
 * @sect Enabling debugging
 *
 * To use dbg in your program you must <code> #include &lt;dbg.h&gt; </code>
 * and compile with the <code>DBG_ENABLED</code> flag set.
 *
 * If you build without DBG_ENABLED you will have no debugging support (neither
 * constraints nor logging). There is no overhead building a program using
 * these utilities when DBG_ENABLED is not set. Well, actually there might be
 * minimal overhead: there is no overhead when using gcc with a little
 * optimisation (<code>-O3</code>). There is a few bytes overhead with
 * optimisation disabled. (The <code>-O1</code> level leaves almost no
 * overhead.)
 *
 * Either way, the rich debugging support is probably worth a few bytes.
 *
 * Once your program is running, you will want to enable diagnostic
 * levels with @ref dbg::enable, and probably attach an ostream (perhaps
 * <code>cerr</code>) to the diagnostic outputs. See the default states section
 * below for information on the initial state of dbg.
 *
 * Aside:
 *  The standard <code>assert</code> macro is an insideous little devil, a lower
 *  case macro. This library replaces it and builds much richer constraints
 *  in its place.
 *  However, because of it, we have to use an API name dbg::assertion,
 *  not dbg::assert - this makes me really cross, but I can't assume that the
 *  user does not <code> #include &lt;assert.h&gt; </code> when using
 *  <code> &lt;dbg.h&gt; </code>.
 *
 * @sect Using constraints
 *
 * The dbg library constraints are very easy to use. Each debugging utility is
 * documented fully to help you understand how they work. Here are some simple
 * examples of library use for run-time constraint checking:
 * <pre>
 *     void test_dbg()
 *     {
 *         dbg::trace trace(DBG_HERE);
 *
 *         int  i   = 5;
 *         int *ptr = &i;
 *
 *         dbg::assertion(DBG_ASSERTION(i != 6));
 *         dbg::check_ptr(ptr, DBG_HERE);
 *
 *         if (i == 5)
 *         {
 *             return;
 *         }
 *
 *         // Shouldn't get here
 *         dbg::sentinel(DBG_HERE);
 *     }
 * </pre>
 *
 * The constraints provided by dbg are:
 *   @li @ref dbg::assertion         - General purpose assertion
 *                                     (a better <code>assert</code>)
 *   @li @ref dbg::sentinel          - Marker for "shouldn't get here" points
 *   @li @ref dbg::unimplemented     - Marks unimplemented code
 *   @li @ref dbg::check_ptr         - Zero pointer check
 *   @li @ref dbg::check_bounds      - Array bounds checking
 *   @li @ref dbg::post_mem_fun      - Member function post condition
 *   @li @ref dbg::post              - General function post condition
 *   @li @ref dbg::compile_assertion - Compile time assertion
 *
 * You can modify constriant behaviour with:
 *   @li @ref dbg::set_assertion_behaviour - Set how contraints behave
 *   @li @ref dbg::set_assertion_period    - Set up trigger periods
 *
 * See their individual documentation for further details on usage.
 *
 * You can specify whether constraints merely report a warning, cause
 * an exception to be thrown, or immediately abort the program (see
 * @ref dbg::assertion_behaviour).
 *
 * For assertions that may fire many times in a tight loop, there is the
 * facility to time-restrict output (see @ref dbg::set_assertion_period)
 *
 * @sect Using logging
 *
 * All the constraint checking shown above integrates with the dbg library
 * stream logging mechanisms. These logging facilities are open for your use as
 * well.
 *
 * Here is a simlpe example of this:
 * <pre>
 *     dbg::attach_ostream(dbg::info, cout);
 *     // now all 'info' messages go to cout
 *
 *     dbg::out(dbg::info)    << "This is some info I want to print out\n";
 *
 *     dbg::out(dbg::tracing) << dbg::indent()
 *                            << "This is output at 'tracing' level, indented "
 *                            << "to the same level as the current tracing "
 *                            << "indent.\n";
 * </pre>
 *
 * When you build without the DBG_ENABLED flag specified, these logging
 * messages will compile out to nothing.
 *
 * The logging is a very flexible system. You can attach multiple ostreams
 * to any dbg output, so you can easily log to a file and log to the console,
 * for example. The output can be formatted in a number of different ways to
 * suit your needs.
 *
 * The logging mechanisms provide you with the ability to prepend to all
 * diagnostic output a standard prefix (see @ref dbg::set_prefix), and
 * also to add the diagnostic level and current time to the prefix (see
 * @ref dbg::enable_level_prefix and @ref dbg::enable_time_prefix).
 *
 * The logging facilities provide by dbg include:
 *   @li @ref dbg::enable              - Enable/disable activity
 *   @li @ref dbg::out                 - Returns a diagnostic ostream
 *   @li @ref dbg::attach_ostream      - Attach an ostream to diagnostic output
 *   @li @ref dbg::detach_ostream      - Detach an ostream to diagnostic output
 *   @li @ref dbg::trace               - Trace entry/exit points
 *
 * The output formatting utilities include:
 *   @li @ref dbg::set_prefix          - Sets the diagnostic output "margin"
 *   @li @ref dbg::enable_level_prefix - More information in messages
 *   @li @ref dbg::enable_time_prefix  - Prints time in messages
 *
 * @sect Diagnostic sources
 *
 * The dbg library allows you to differentiate different "sources" of logging.
 *
 * Each of the debug utilities has a second form in which you can supply
 * a string describing the source of the diagnostic output (see
 * @ref dbg::dbg_source). This source may be a different software component, a
 * separate file - whatever granularity you like!
 *
 * If you don't specify a @ref dbg::dbg_source then you are working with the
 * ordinary "unnamed" source.
 *
 * Using these forms you can filter out diagnostics from the different
 * parts of your code. Each source can also be attached to a different set of
 * streams (logging each component to a separate file, for example). The
 * filtering is rich - you can selectively filter each different diagnostic
 * @ref dbg::level for each @ref dbg::dbg_source. For example,
 *
 * <pre>
 *     dbg::enable(dbg::all, "foo-driver", true);
 *     dbg::enable(dbg::all, "bar-driver", false);
 *
 *     int i = 5;
 *     dbg::assertion("foo-driver", DBG_ASSERTION(i != 6));
 *     dbg::assertion("bar-driver", DBG_ASSERTION(i != 6));
 * </pre>
 *
 * This will trigger an assertion for the "foo-driver" but not the
 * "bar-driver".
 *
 * There is no requirement to "register" a @ref dbg::dbg_source. The first
 * time you use it in any of the dbg APIs, it will be registered with the dbg
 * library. It comes into an existance as a copy of the "default"
 * debugging sourcei, @ref dbg::default_source.
 *
 * The default source initially has all debug levels disabled.
 * You can change that with this call. Note that this function
 * only affects sources created <i>after</i> the call is made.
 * Existing sources are unaffected.
 *
 * If you don't know all of the @ref dbg::dbg_source sources currently
 * available, you can blanket enable/disable them with @ref dbg::enable_all.
 *
 * It can be tedious to specify the @ref dbg_source in every dbg call in
 * a source file. For this reason, you can specify the DBG_SOURCE compile
 * time macro (wherever you specify DBG_ENABLED). When set, the calls
 * automatically recieve the source name via the DBG_HERE macro (see
 * @ref dbg::source_pos for details). If DBG_SOURCE is supplied but you call
 * a dbg API with a specific named @ref dbg_source, this name will override
 * the underlying DBG_SOURCE name.
 *
 * @sect Overloads
 *
 * Each constraint utility has a number of overloaded forms. This is to make
 * using them more convenient. The most rich overload allows you to specify
 * a diagnostic @ref dbg::level and a @ref dbg::dbg_source. There are other
 * versions that omit one of these parameters, assuming a relevant default.
 *
 * @sect Default states
 *
 * When your program first starts up the dbg library has all debugging levels
 * switched off. You can enable debugging with @ref dbg::enable. All of the
 * possible @ref dbg::dbg_source enables are also all off for all
 * levels. You can enable these with @ref dbg::enable, or @ref dbg::enable_all.
 *
 * Initially, the <code>std::cerr</code> stream is attached to the
 * @ref dbg::error and @ref dbg::fatal diagnostic levels. You can
 * attach ostreams to the other diagnostic levels with @ref
 * dbg::attach_ostream.
 *
 * You can modify the "default state" of newly created debug sources. To do
 * this use the special @ref dbg::default_source source name in calls to
 * @ref dbg::enable, @ref dbg::attach_ostream, and and @ref detach_ostream.
 * New sources take the setup from this template source.
 *
 * All assertion levels are set to @ref dbg::assertions_abort at first, like
 * the standard library's assert macro. You can change this behaviour with
 * @ref dbg::set_assertion_behaviour. There are no timeout periods set - you
 * can change this with @ref dbg::set_assertion_period.
 *
 * @short   Debugging utilities
 * @author  Pete Goodliffe
 * @version 1.0
 */
namespace dbg {
  /**
   * This is the version number of the dbg library.
   *
   * The value is encoded as version * 100. This means that 100 represents
   * version 1.00, for example.
   */
  const int version = 110;

  /**************************************************************************
   * Debugging declarations
   *************************************************************************/

  /**
   * The various predefined debugging levels. The dbg API calls use these
   * levels as parameters, and allow the user to sift the less interesting
   * debugging levels out through @ref dbg::enable.
   *
   * These levels (and their intended uses) are:
   *   @li info    - Informational, just for interest
   *   @li warning - For warnings, bad things but recoverable
   *   @li error   - For errors that can't be recovered from
   *   @li fatal   - Errors at this level will cause the dbg library to abort
   *                 program execution, no matter what the
   *                 @ref assertion_behaviour is set to
   *   @li tracing - Program execution tracing messages
   *   @li debug   - Messages about the state of dbg library, you cannot
   *                 generate messages at this level
   *   @li none    - For APIs that use 'no level specified'
   *   @li all     - Used in @ref enable and @ref attach_ostream to
   *                 specify all levels
   */
  enum level {
    info,
    warning,
    error,
    fatal,
    tracing,
    debug,
    none,
    all
  };

  /**
   * This enum type describes what happens when a debugging assertion
   * fails. The behaviour can be:
   *   @li assertions_abort    - Assertions cause a program abort
   *   @li assertions_throw    - Assertions cause a @ref dbg_exception to
   *                             be thrown
   *   @li assertions_continue - Assertions cause the standard diagnostic
   *                             printout to occur (the same as the above
   *                             behaviours) but execution continues
   *                             regardless
   *
   * The dbg library defaults to assertions_abort behaviour, like the
   * standard C <code>assert</code>.
   *
   * @see dbg::set_assertion_behaviour
   */
  enum assertion_behaviour {
    assertions_abort,
    assertions_throw,
    assertions_continue
  };

  /**
   * typedef for a string that describes the "source" of a diagnostic. If
   * you are working on a large project with many small code modules you may
   * only want to enable debugging from particular source modules. This
   * typedef facilitiates this.
   *
   * Depending on the desired granularity of your dbg sources you will use
   * different naming conventions. For example, your dbg_sources might
   * be filenames, that way you can switch off all debugging output from
   * a particular file quite easily. It might be device driver names,
   * component names, library names, or even function names. It's up to you.
   *
   * If you provide the DBG_SOURCE macro definition at compile time, then
   * the DBG_HERE macro includes this source name, differentiating the
   * sources for you automatically.
   *
   * @see dbg::enable(level,dbg_source,bool)
   * @see dbg::enable_all
   */
  typedef const char * dbg_source;

  /**************************************************************************
   * source_pos
   *************************************************************************/

  /**
   * Typedef used in the @ref source_pos data structure.
   *
   * Describes a line number in a source file.
   *
   * @see dbg::source_pos
   */
  typedef const unsigned int line_no_t;

  /**
   * Typedef used in the @ref source_pos data structure.
   *
   * Describes a function name in a source file. (Can be zero to
   * indicate the function name cannot be assertained on this compiler).
   *
   * @see dbg::source_pos
   */
  typedef const char * func_name_t;

  /**
   * Typedef used in the @ref source_pos data structure.
   *
   * Describes a filename.
   *
   * @see dbg::source_pos
   */
  typedef const char * file_name_t;

  /**
   * Data structure describing a position in the source file. That is,
   *   @li The line number
   *   @li The function name (if the compiler supports this)
   *   @li The filename
   *   @li The @ref dbg_soruce specified by DBG_SOURCE compilation
   *       parameter, if any (otherwise zero)
   *
   * To create a source_pos for the current position, you can use
   * the DBG_HERE convenience macro.
   *
   * There is an empty constructor that allows you to create a source_pos
   * that represents 'no position specified'.
   *
   * This structure should only be used in dbg library API calls.
   *
   * You can print a source_pos using the usual stream manipulator syntax.
   */
  struct source_pos {
    line_no_t   line;
    func_name_t func;
    file_name_t file;
    dbg_source  src;

    /**
     * Creates a source_pos struct. Use the DBG_HERE macro to
     * call this constructor conveniently.
     */
    source_pos(line_no_t ln, func_name_t fn, file_name_t fl, dbg_source s)
      : line(ln), func(fn), file(fl), src(s) {}

    /**
     * A 'null' source_pos for 'no position specified'
     */
    source_pos()
      : line(0), func(0), file(0), src(0) {}
  };

#ifndef _MSC_VER
  /**
   * The dbgclock_t typedef is an unfortunate workaround for comptability
   * purposes. One (unnamed) popular compiler platform supplies a
   * <ctime> header file, but this header does NOT place the contents
   * into the std namespace.
   *
   * This typedef is the most elegant work around for that problem. It is
   * conditionally set to the appropriate clock_t definition.
   *
   * In an ideal world this would not exist.
   *
   * This is the version for sane, standards-compliant platforms.
   */
  typedef std::clock_t dbgclock_t;
#else
  /**
   * See dbgclock_t documentation above. This is the version for broken
   * compiler platforms.
   */
  typedef clock_t dbgclock_t;
#endif

  /**************************************************************************
   * Exceptions
   *************************************************************************/

  /**
   * The base type of exception thrown by dbg assertions (and other dbg
   * library constraint checks) if the @ref assertion_behaviour is set to
   * assertions_throw.
   *
   * The exception keeps a record of the source position of the trigger
   * for this exception.
   */
  struct dbg_exception : public std::exception {
    dbg_exception(const source_pos &p) : pos(p) {}
    const source_pos pos;
  };

  /**
   * The type of exception thrown by @ref assertion.
   *
   * @see assertion
   */
  struct assertion_exception : public dbg_exception {
    assertion_exception(const source_pos &p) : dbg_exception(p) {}
  };

  /**
   * The type of exception thrown by @ref sentinel.
   *
   * @see sentinel
   */
  struct sentinel_exception : public dbg_exception {
    sentinel_exception(const source_pos &p) : dbg_exception(p) {}
  };

  /**
   * The type of exception thrown by @ref unimplemented.
   *
   * @see unimplemented
   */
  struct unimplemented_exception : public dbg_exception {
    unimplemented_exception(const source_pos &p) : dbg_exception(p) {}
  };

  /**
   * The type of exception thrown by @ref check_ptr.
   *
   * @see check_ptr
   */
  struct check_ptr_exception : public dbg_exception {
    check_ptr_exception(const source_pos &p) : dbg_exception(p) {}
  };

#ifdef DBG_ENABLED

  /**************************************************************************
   * default_source
   *************************************************************************/

  /**
   * The name of a "template" debugging source that provides the default
   * state for newly created sources. You can attach and detach logging
   * streams here, and enable/disable logging levels.
   *
   * All source state is copied from the default_source to a new dbg_source.
   *
   * Whilst you can also use this source for diagnostic purposes this isn't
   * it's intention, and it would be confusing to do so.
   *
   * See @ref dbg_source for discussion on the use of debugging sources in
   * dbg.
   *
   * @see dbg_source
   */
  extern dbg_source default_source;

  /**************************************************************************
   * Debug version of the DBG_HERE macro
   *************************************************************************/

  /*
   * DBG_FUNCTION is defined to be a macro that expands to the name of
   * the current function, or zero if the compiler is unable to supply that
   * information. It's sad that this wasn't included in the C++ standard
   * from the very beginning.
   */
#if defined(__GNUC__)
#define DBG_FUNCTION __FUNCTION__
#else
#define DBG_FUNCTION 0
#endif

#if !defined(DBG_SOURCE)
#define DBG_SOURCE 0
#endif

  /*
   * Handy macro to generate a @ref source_pos object containing the
   * information of the current source line.
   *
   * @see dbg::source_pos
   */
#define DBG_HERE \
        (::dbg::source_pos(__LINE__, DBG_FUNCTION, __FILE__, DBG_SOURCE))

  /**************************************************************************
   * Enable/disable dbg facilities
   *************************************************************************/

  /**
   * Enables or disables a particular debugging level. The affects dbg
   * library calls which don't specify a @ref dbg_source, i.e. from the
   * unnamed source.
   *
   * Enabling affects both constraint checking and diagnostic log output.
   *
   * If you enable a debugging level twice you only need to disable it once.
   *
   * All diagnostic output is initially disabled. You can easily enable
   * output in your main() thus:
   * <pre>
   *     dbg::enable(dbg::all, true);
   * </pre>
   *
   * Note that if dbg library calls do specify a @ref dbg_source, or you
   * provide a definition for the DBG_SOURCE macro on compilation, then you
   * will instead need to enable output for that particular source. Use the
   * overloaded version of enable. This version of enable doesn't affect
   * these other @ref dbg_source calls.
   *
   * @param lvl     Diagnostic level to enable/disable
   * @param enabled true to enable this diagnostic level, false to disable it
   * @see   dbg::enable_all
   * @see   dbg::out
   * @see   dbg::attach_ostream
   */
  void enable(level lvl, bool enabled);

  /**
   * In addition to the above enable function, this overloaded version is
   * used when you use dbg APIs with a @ref dbg_source specified. For these
   * versions of the APIs no debugging will be performed unless you
   * enable it with this API.
   *
   * To enable debugging for the "foobar" diagnostic source at the info
   * level you need to do the following:
   * <pre>
   *     dbg::enable(dbg::info, "foobar", true);
   * </pre>
   *
   * If you enable a level for a particular @ref dbg_source twice you only
   * need to disable it once.
   *
   * @param lvl     Diagnostic level to enable/disable for the @ref dbg_source
   * @param src     String describing the diagnostic source
   * @param enabled true to enable this diagnostic level, false to disable it
   * @see   dbg::out
   */
  void enable(level lvl, dbg_source src, bool enabled);

  /**
   * You may not know every single @ref dbg_source that is generating
   * debugging in a particular code base. However, using this function
   * you can enable a diagnostic level for all currently registered sources
   * in one fell swoop.
   *
   * For example,
   * <pre>
   *     dbg::enable_all(dbg::all, true);
   * </pre>
   */
  void enable_all(level lvl, bool enabled);

  /**************************************************************************
   * Logging
   *************************************************************************/

  /**
   * Returns an ostream suitable for sending diagnostic messages to.
   * Each diagnostic level has a different logging ostream which can be
   * enabled/disabled independantly. In addition, each @ref dbg_source
   * has separate enables/disables for each diagnostic level.
   *
   * This overloaded version of out is used when you are creating diagnostics
   * that are tied to a particular @ref dbg_source.
   *
   * It allows you to write code like this:
   * <pre>
   *     dbg::out(dbg::info, "foobar") << "The foobar is flaky\n";
   * </pre>
   *
   * If you want to prefix your diagnostics with the standard dbg library
   * prefix (see @ref set_prefix) then use the @ref prefix or @ref indent
   * stream manipulators.
   *
   * @param lvl Diagnostic level get get ostream for
   * @param src String describing the diagnostic source
   */
  std::ostream &out(level lvl, dbg_source src);

  /**
   * Returns an ostream suitable for sending diagnostic messages to.
   * Each diagnostic level has a different logging ostream which can be
   * enabled/disabled independantly.
   *
   * You use this version of out when you are creating diagnostics
   * that aren't tidied to a particular @ref dbg_source.
   *
   * Each diagnostic @ref dbg_source has a separate set of streams.
   * This function returns the stream for the "unnamed" source. Use the
   * overload below to obtain the stream for a named source.
   *
   * It allows you to write code like this:
   * <pre>
   *     dbg::out(dbg::info) << "The code is flaky\n";
   * </pre>
   *
   * If you want to prefix your diagnostics with the standard dbg library
   * prefix (see @ref set_prefix) then use the @ref prefix or @ref indent
   * stream manipulators.
   *
   * @param lvl Diagnostic level get get ostream for
   */
  inline std::ostream &out(level lvl) {
    return out(lvl, 0);
  }

  /**
   * Attaches the specified ostream to the given diagnostic level
   * for the "unnamed" debug source. Now when diagnostics are produced
   * at that level, this ostream will recieve a copy.
   *
   * You can attach multiple ostreams to a diagnostic level. Be careful
   * that they don't go to the same place (e.g. cout and cerr both going
   * to your console) - this might confuse you!
   *
   * If you attach a ostream mutiple times it will only receive one
   * copy of the diagnostics, and you will only need to call
   * @ref detach_ostream once.
   *
   * Remember, don't destroy the ostream without first removing it from
   * dbg libary, or Bad Things will happen.
   *
   * @param lvl Diagnostic level
   * @param o   ostream to attach
   * @see   dbg::detach_ostream
   * @see   dbg::detach_all_ostreams
   */
  void attach_ostream(level lvl, std::ostream &o);

  /**
   * Attaches the specified ostream to the given diagnostic level
   * for the specified debug source. Otherwise, similar to
   * @ref dbg::attach_ostream above.
   *
   * @param lvl  Diagnostic level
   * @param src  Debug source
   * @param o    ostream to attach
   * @see   dbg::detach_ostream
   * @see   dbg::detach_all_ostreams
   */
  void attach_ostream(level lvl, dbg_source src, std::ostream &o);

  /**
   * Detaches the specified ostream from the given diagnostic level.
   *
   * If the ostream was not attached then no error is generated.
   *
   * If you attached the ostream twice, one call to detach_ostream will
   * remove it completely.
   *
   * @param lvl Diagnostic level
   * @param o   ostream to detach
   * @see   dbg::attach_ostream
   * @see   dbg::detach_all_ostreams
   */
  void detach_ostream(level lvl, std::ostream &o);

  /**
   * Detaches the specified ostream from the given diagnostic level
   * for the specified debug source. Otherwise, similar to
   * @ref dbg::detach_ostream above.
   *
   * @param lvl Diagnostic level
   * @param src Debug source
   * @param o   ostream to detach
   * @see   dbg::attach_ostream
   * @see   dbg::detach_all_ostreams
   */
  void detach_ostream(level lvl, dbg_source src, std::ostream &o);

  /**
   * Detaches all attached ostreams from the specified diagnostic level
   * for the "unnamed" diagnostic source.
   *
   * @param lvl Diagnostic level
   * @see   dbg::attach_ostream
   * @see   dbg::detach_ostream
   */
  void detach_all_ostreams(level lvl);

  /**
   * Detaches all attached ostreams from the specified diagnostic level
   * for the specified debug source. Otherwise, similar to
   * @ref dbg::detach_all_ostreams above.
   *
   * @param lvl Diagnostic level
   * @see   dbg::attach_ostream
   * @see   dbg::detach_ostream
   */
  void detach_all_ostreams(level lvl, dbg_source src);

  /**
   * Convenience function that returns the ostream for the info
   * @ref dbg::level for the "unnamed" source.
   *
   * @see dbg::out
   */
  inline std::ostream &info_out() {
    return out(dbg::info);
  }

  /**
   * Convenience function that returns the ostream for the warning
   * @ref dbg::level for the "unnamed" source.
   *
   * @see dbg::out
   */
  inline std::ostream &warning_out() {
    return out(dbg::warning);
  }

  /**
   * Convenience function that returns the ostream for the error
   * @ref dbg::level for the "unnamed" source.
   *
   * @see dbg::out
   */
  inline std::ostream &error_out() {
    return out(dbg::error);
  }

  /**
   * Convenience function that returns the ostream for the fatal
   * @ref dbg::level for the "unnamed" source.
   *
   * @see dbg::out
   */
  inline std::ostream &fatal_out() {
    return out(dbg::fatal);
  }

  /**
   * Convenience function that returns the ostream for the tracing
   * @ref dbg::level for the "unnamed" source.
   *
   * @see dbg::out
   */
  inline std::ostream &trace_out() {
    return out(dbg::tracing);
  }

  /**************************************************************************
   * Output formatting
   *************************************************************************/

  /**
   * Sets the debugging prefix - the characters printed before any
   * diagnostic output. Defaults to "*** ".
   *
   * @param prefix New prefix string
   * @see   dbg::prefix
   * @see   dbg::enable_level_prefix
   * @see   dbg::enable_time_prefix
   */
  void set_prefix(const char *prefix);

  /**
   * The dbg library can add to the @ref prefix the name of the used
   * diagnostic level (e.g. info, fatal, etc).
   *
   * By default, this facility is disabled. This function allows you to
   * enable the facility.
   *
   * @param enabled true to enable level prefixing, false to disable
   * @see   dbg::set_prefix
   * @see   dbg::enable_time_prefix
   */
  void enable_level_prefix(bool enabled);

  /**
   * The dbg library can add to the @ref prefix the current time. This
   * can be useful when debugging systems which remain active for long
   * periods of time.
   *
   * By default, this facility is disabled. This function allows you to
   * enable the facility.
   *
   * The time is produced in the format of the standard library ctime
   * function.
   *
   * @param enabled true to enable time prefixing, false to disable
   * @see   dbg::set_prefix
   * @see   dbg::enable_level_prefix
   */
  void enable_time_prefix(bool enabled);

  /**
   * Used so that you can produce a prefix in your diagnostic output in the
   * same way that the debugging library does.
   *
   * You can use it in one of two ways: with or without a diagnostic
   * @ref level. For the latter, if level prefixing is enabled (see
   * @ref enable_level_prefix) then produces a prefix including the
   * specified diagnostic level text.
   *
   * Examples of use:
   *
   * <pre>
   *     dbg::out(dbg::info) << dbg::prefix()
   *                         << "A Bad Thing happened\n";
   *
   *     dbg::out(dbg::info) << dbg::prefix(dbg::info)
   *                         << "A Bad Thing happened\n";
   * </pre>
   *
   * @see dbg::indent
   * @see dbg::set_prefix
   * @see dbg::enable_level_prefix
   * @see dbg::enable_time_prefix
   */
  struct prefix {
    /**
     * Creates a prefix with no specified diagnostic @ref level.
     * No diagnostic level text will be included in the prefix.
     */
    prefix() : l(none) {}

    /**
     * @param lvl Diagnostic @ref level to include in prefix
     */
    prefix(level lvl) : l(lvl) {}

    level l;
  };

  /**
   * This is called when you use the @ref prefix stream manipulator.
   *
   * @internal
   * @see dbg::prefix
   */
  std::ostream &operator<<(std::ostream &s, const prefix &p);

  /**
   * Used so that you can indent your diagnostic output to the same level
   * as the debugging library. This also produces the @ref prefix output.
   *
   * Examples of use:
   *
   * <pre>
   *     dbg::out(dbg::info) << dbg::indent()
   *                         << "A Bad Thing happened\n";
   *
   *     dbg::out(dbg::info) << dbg::indent(dbg::info)
   *                         << "A Bad Thing happened\n";
   * </pre>
   *
   * @see dbg::prefix
   * @see dbg::set_prefix
   * @see dbg::enable_level_prefix
   * @see dbg::enable_time_prefix
   */
  struct indent {
    /**
     * Creates a indent with no specified diagnostic @ref level.
     * No diagnostic level text will be included in the @ref prefix part.
     */
    indent() : l(none) {}

    /**
     * @param lvl Diagnostic level to include in prefix
     */
    indent(level lvl) : l(lvl) {}

    level l;
  };

  /**
   * This is called when you use the @ref indent stream manipulator.
   *
   * @internal
   * @see dbg::indent
   */
  std::ostream &operator<<(std::ostream &s, const indent &i);

  /**
   * This is called when you send a @ref source_pos to a diagnostic output.
   * You can use this to easily check the flow of execcution in your
   * program.
   *
   * For example,
   * <pre>
   *     dbg::out(dbg::tracing) << DBG_HERE << std::endl;
   * </pre>
   *
   * Take care that you only send DBG_HERE to the diagnostic outputs
   * (obtained with @ref dbg::out) and not "ordinary" streams like
   * <code>std::cout</code>.
   *
   * In non debug builds, DBG_HERE is a "no-op" doing nothing, and so no
   * useful output will be produced on cout.
   *
   * @internal
   * @see dbg::indent
   */
  std::ostream &operator<<(std::ostream &s, const source_pos &pos);

  /**************************************************************************
   * Behaviour
   *************************************************************************/

  /**
   * Sets what happens when assertions (or other constraints) trigger. There
   * will always be diagnostic ouput. Assertions have 'abort' behaviour by
   * default - like the ISO C standard, they cause an abort.
   *
   * If an assertion is encountered at the fatal level, the debugging library
   * will abort the program regardless of this behaviour setting.
   *
   * If a diagnostic level is not enabled (see @ref enable) then the
   * @ref assertion_behaviour is not enacted, and no output is produced.
   *
   * @param lvl       Diagnostic level to set behaviour for
   * @param behaviour Assertion behaviour
   * @see   dbg::set_assertion_period
   * @see   dbg::enable
   * @see   dbg::assertion
   * @see   dbg::sentinel
   * @see   dbg::unimplemented
   * @see   dbg::check_ptr
   */
  void set_assertion_behaviour(level lvl, assertion_behaviour behaviour);

  /**
   * You may want an assertion to trigger once only and then for subsequent
   * calls to remain inactive. For example, if there is an @ref assertion in
   * a loop you may not want diagnostics produced for each loop iteration.
   *
   * To do this, you do the following:
   * <pre>
   *      // Prevent several thousand diagnostic print outs
   *      dbg::set_assertion_period(CLOCKS_PER_SEC);
   *
   *      // Example loop
   *      int array[LARGE_VALUE];
   *      put_stuff_in_array(array);
   *      for(unsigned int n = 0; n < LARGE_VALUE; n++)
   *      {
   *          dbg::assertion(DBG_ASSERT(array[n] != 0));
   *          do_something(array[n]);
   *      }
   * </pre>
   *
   * set_assertion_period forces a certain time period between triggers of a
   * particular constraint. The @ref assertion in the example above will only
   * be triggered once a second (despite the fact that the constraint
   * condition will be broken thousands of times a second). This will not
   * affect any other @ref assertion - they will each have their own timeout
   * periods.
   *
   * Setting a period of zero disables any constraint period.
   *
   * The default behaviour is to have no period.
   *
   * If a period is set then diagnostic printouts will include the number
   * of times each constraint has been triggered (since the period was set).
   * Using this, even if diagnostics don't always appear on the attached
   * ostreams you have some indication of how often each constraint is
   * triggered.
   *
   * This call only really makes sense if the @ref assertion_behaviour is
   * set to @ref assertions_continue.
   *
   * @param period Time between triggerings of each assertion, or zero to
   *               disable
   * @see   dbg::set_assertion_behaviour
   * @see   dbg::assertion
   * @see   dbg::sentinel
   * @see   dbg::unimplemented
   * @see   dbg::check_ptr
   */
  void set_assertion_period(dbgclock_t period);

  /**************************************************************************
   * Assertion
   *************************************************************************/

  /**
   * Describes an @ref assertion.
   *
   * This is an internal data structure, you do not need to create it
   * directly. Use the DBG_ASSERTION macro to create it.
   *
   * @internal
   * @see dbg::assertion
   */
  struct assert_info : public source_pos {
    bool        asserted;
    const char *text;

    /**
     * Do not call this directly. Use the DBG_ASSERTION macro.
     *
     * @internal
     */
    assert_info(bool a, const char *t,
                line_no_t line, func_name_t func,
                file_name_t file, dbg_source spos)
      : source_pos(line, func, file, spos), asserted(a), text(t) {}

    /**
     * Do not call this directly. Use the DBG_ASSERTION macro.
     *
     * @internal
     */
    assert_info(bool a, const char *b, const source_pos &sp)
      : source_pos(sp), asserted(a), text(b) {}
  };

  /*
   * Utility macro used by the DBG_ASSERTION macro - it converts a
   * macro parameter into a character string.
   */
#define DBG_STRING(a) #a

  /*
   * Handy macro used by clients of the @ref dbg::assertion function.
   * It use is described in the @ref assertion documentation.
   *
   * @see dbg::assertion
   */
#define DBG_ASSERTION(a) \
        ::dbg::assert_info(a, DBG_STRING(a), DBG_HERE)

  // PATCH by mandor
  void init();

  /**
   * Used to assert a constraint in your code. Use the DBG_ASSERTION macro
   * to generate the third parameter.
   *
   * This version creates an assertion bound to a particular @ref dbg_source.
   *
   * The assertion is the most general constraint utility - there are others
   * which have more specific purposes (like @ref check_ptr to ensure a
   * pointer is non-null). assertion allows you to test any boolean
   * expression.
   *
   * To use assertion for a @ref dbg_source "foobar" you write code like:
   * <pre>
   *     int i = 0;
   *     dbg::assertion(info, "foobar", DBG_ASSERTION(i != 0));
   * </pre>
   *
   * If you build with debugging enabled (see @ref dbg) the program will
   * produce diagnostic output to the relevant output stream if the
   * constraint fails, and the appropriate @ref assertion_behaviour
   * is enacted.
   *
   * Since in non-debug builds the expression in the DBG_ASSERTION macro
   * will not be evaluated, it is important that the expression has no
   * side effects.
   *
   * @param lvl Diagnostic level to assert at
   * @param src String describing the diagnostic source
   * @param ai  assert_info structure created with DBG_ASSERTION
   */
  void assertion(level lvl, dbg_source src, const assert_info &ai);

  /**
   * Overloaded version of @ref assertion that is not bound to a particular
   * @ref dbg_source.
   *
   * @param lvl Diagnostic level to assert at
   * @param ai  assert_info structure created with DBG_ASSERTION
   */
  inline void assertion(level lvl, const assert_info &ai) {
    assertion(lvl, 0, ai);
  }

  /**
   * Overloaded version of @ref assertion that defaults to the
   * warning @ref level.
   *
   * @param src String describing the diagnostic source
   * @param ai assert_info structure created with DBG_ASSERTION
   */
  inline void assertion(dbg_source src, const assert_info &ai) {
    assertion(warning, src, ai);
  }

  /**
   * Overloaded version of @ref assertion that defaults to the
   * warning @ref level and is not bound to a particular @ref dbg_source.
   *
   * @param ai assert_info structure created with DBG_ASSERTION
   */
  inline void assertion(const assert_info &ai) {
    assertion(warning, 0, ai);
  }

  /**************************************************************************
   * Sentinel
   *************************************************************************/

  /**
   * You should put this directly after a "should never get here" comment.
   *
   * <pre>
   *      int i = 5;
   *      if (i == 5)
   *      {
   *          std::cout << "Correct program behaviour\n";
   *      }
   *      else
   *      {
   *          dbg::sentinel(dbg::error, "foobar", DBG_HERE);
   *      }
   * </pre>
   *
   * @param lvl  Diagnostic level to assert at
   * @param src  String describing the diagnostic source
   * @param here Supply DBG_HERE
   */
  void sentinel(level lvl, dbg_source src, const source_pos &here);

  /**
   * Overloaded version of @ref sentinel that is not bound to a particular
   * @ref dbg_source.
   *
   * @param lvl  Diagnostic level to assert at
   * @param here Supply DBG_HERE
   */
  inline void sentinel(level lvl, const source_pos &here) {
    sentinel(lvl, 0, here);
  }

  /**
   * Overloaded version of @ref sentinel that defaults to the warning
   * @ref level and is not bound to a particular @ref dbg_source.
   *
   * @param src  String describing the diagnostic source
   * @param here Supply DBG_HERE
   */
  inline void sentinel(dbg_source src, const source_pos &here) {
    sentinel(warning, src, here);
  }

  /**
   * Overloaded version of @ref sentinel that defaults to the warning
   * @ref level and is not bound to a particular @ref dbg_source.
   *
   * @param here Supply DBG_HERE
   */
  inline void sentinel(const source_pos &here) {
    sentinel(warning, 0, here);
  }

  /**************************************************************************
   * Unimplemented
   *************************************************************************/

  /**
   * You should put this directly after a "this has not been implemented
   * (yet)" comment.
   *
   * <pre>
   *      switch (variable)
   *      {
   *          ...
   *          case SOMETHING:
   *          {
   *              dbg::unimplemented(dbg::warning, "foobar", DBG_HERE);
   *              break;
   *          }
   *          ...
   *      }
   * </pre>
   *
   * Note the "break;" above - if the @ref assertion_behaviour is non-fatal
   * then execution will continue. You wouldn't want unintentional
   * fall-through.
   *
   * @param lvl  Diagnostic level to assert at
   * @param src  String describing the diagnostic source
   * @param here Supply DBG_HERE
   *
   */
  void unimplemented(level lvl, dbg_source src, const source_pos &here);

  /**
   * Overloaded version of @ref unimplemented that is not bound to a
   * particular @ref dbg_source.
   *
   * @param lvl  Diagnostic level to assert at
   * @param here Supply DBG_HERE
   */
  inline void unimplemented(level lvl, const source_pos &here) {
    unimplemented(lvl, 0, here);
  }

  /**
   * Overloaded version of @ref unimplemented that defaults to the
   * warning @ref level.
   *
   * @param src  String describing the diagnostic source
   * @param here Supply DBG_HERE
   */
  inline void unimplemented(dbg_source src, const source_pos &here) {
    unimplemented(warning, src, here);
  }

  /**
   * Overloaded version of @ref unimplemented that defaults to the
   * warning @ref level and is not bound to a particular @ref dbg_source.
   *
   * @param here Supply DBG_HERE
   */
  inline void unimplemented(const source_pos &here) {
    unimplemented(warning, 0, here);
  }

  /**************************************************************************
   * Pointer checking
   *************************************************************************/

  /**
   * A diagnostic function to assert that a pointer is not zero.
   *
   * To use it you write code like:
   * <pre>
   *     void *p = 0;
   *     dbg::check_ptr(dbg::info, "foobar", p, DBG_HERE);
   * </pre>
   *
   * It's better to use this than a general purpose @ref assertion. It
   * reads far more intuitively in your code.
   *
   * @param lvl  Diagnostic level to assert at
   * @param src  String describing the diagnostic source
   * @param p    Pointer to check
   * @param here Supply DBG_HERE
   */
  void check_ptr(level lvl, dbg_source src, void *p, const source_pos &here);

  /**
   * Overloaded version of @ref check_ptr that is not bound to a particular
   * @ref dbg_source.
   *
   * @param lvl  Diagnostic level to assert at
   * @param p    Pointer to check
   * @param here Supply DBG_HERE
   */
  inline void check_ptr(level lvl, void *p, const source_pos &here) {
    check_ptr(lvl, 0, p, here);
  }

  /**
   * Overloaded version of @ref check_ptr that defaults to the
   * warning @ref level.
   *
   * @param src  String describing the diagnostic source
   * @param p    Pointer to check
   * @param here Supply DBG_HERE
   */
  inline void check_ptr(dbg_source src, void *p, const source_pos &here) {
    check_ptr(warning, src, p, here);
  }

  /**
   * Overloaded version of @ref check_ptr that defaults to the
   * warning @ref level and is not bound to a particular @ref dbg_source.
   *
   * @param p    Pointer to check
   * @param here Supply DBG_HERE
   */
  inline void check_ptr(void *p, const source_pos &here) {
    check_ptr(warning, 0, p, here);
  }

  /**************************************************************************
   * Bounds checking
   *************************************************************************/

  /**
   * Utility that determines the number of elements in an array. Used
   * by the @ref check_bounds constraint utility function.
   *
   * This is not available in non-debug versions, so do not use it
   * directly.
   *
   * @param  array Array to determine size of
   * @return The number of elements in the array
   * @internal
   */
  template <class T>
  inline unsigned int array_size(T &array) {
    return sizeof(array)/sizeof(array[0]);
  }

  /**
   * A diagnostic function to assert that an array access is not out
   * of bounds.
   *
   * You probably want to use the more convenient check_bounds versions
   * below if you are accessing an array whose definition is in scope -
   * the compiler will then safely detrmine the size of the array for you.
   *
   * @param lvl   Diagnostic level to assert at
   * @param src   String describing the diagnostic source
   * @param index Test index
   * @param bound Boundary value (index must be < bound, and >= 0)
   * @param here  Supply DBG_HERE
   */
  void check_bounds(level lvl, dbg_source src,
                    int index, int bound, const source_pos &here);
  /**
   * A diagnostic function to assert that an array access is not out
   * of bounds. With this version you can specify the minimum and maximum
   * bound value.
   *
   * You probably want to use the more convenient check_bounds version
   * below if you are accessing an array whose definition is in scope -
   * the compiler will then safely detrmine the size of the array for you.
   *
   * @param lvl      Diagnostic level to assert at
   * @param src      String describing the diagnostic source
   * @param index    Test index
   * @param minbound Minimum bound (index must be >= minbound
   * @param maxbound Minimum bound (index must be < maxbound)
   * @param here     Supply DBG_HERE
   */
  inline void check_bounds(level lvl, dbg_source src,
                           int index, int minbound, int maxbound,
                           const source_pos &here) {
    check_bounds(lvl, src, index-minbound, maxbound, here);
  }

  /**
   * Overloaded version of check_bounds that can automatically determine the
   * size of an array if it within the current scope.
   *
   * You use it like this:
   * <pre>
   *     int a[10];
   *     int index = 10;
   *     dbg::check_bounds(dbg::error, index, a, DBG_HERE);
   *     a[index] = 5;
   * </pre>
   *
   * @param lvl   Diagnostic level to assert at
   * @param src  String describing the diagnostic source
   * @param index Test index
   * @param array Array index is applied to
   * @param here  Supply DBG_HERE
   */
  template <class T>
  void check_bounds(level lvl, dbg_source src,
                    int index, T &array, const source_pos &here) {
    check_bounds(lvl, src, index, array_size(array), here);
  }

  /**
   * Overloaded version of @ref check_bounds that is not bound to a
   * particular @ref dbg_source.
   *
   * @param lvl   Diagnostic level to assert at
   * @param index Test index
   * @param array Array index is applied to
   * @param here  Supply DBG_HERE
   */
  template <class T>
  void check_bounds(level lvl, int index, T &array, const source_pos &here) {
    check_bounds(lvl, 0, index, array_size(array), here);
  }

  /**
   * Overloaded version of @ref check_bounds that defaults to the
   * warning @ref level.
   *
   * @param src   String describing the diagnostic source
   * @param index Test index
   * @param array Array index is applied to
   * @param here  Supply DBG_HERE
   */
  template <class T>
  void check_bounds(dbg_source src, int index, T &array,
                    const source_pos &here) {
    check_bounds(warning, src, index, array_size(array), here);
  }

  /**
   * Overloaded version of @ref check_bounds that defaults to the
   * warning @ref level and is not bound to a particular @ref dbg_source.
   *
   * @param index Test index
   * @param array Array index is applied to
   * @param here  Supply DBG_HERE
   */
  template <class T>
  void check_bounds(int index, T &array, const source_pos &here) {
    check_bounds(warning, 0, index, array_size(array), here);
  }

  /**************************************************************************
   * Tracing
   *************************************************************************/

  /**
   * The trace class allows you to easily produce tracing diagnostics.
   *
   * When the ctor is called, it prints "->" and the name of the
   * function, increasing the indent level. When the object is deleted
   * it prints "<-" followed again by the name of the function.
   *
   * You can use the name of the current function gathered via the
   * DBG_HERE macro, or some other tracing string you supply.
   *
   * Diagnostics are produced at the tracing @ref level.
   *
   * For example, if you write the following code:
   *
   * <pre>
   *     void foo()
   *     {
   *         dbg::trace t1(DBG_HERE);
   *         // do some stuff
   *         {
   *             dbg::trace t2("sub block");
   *             // do some stuff
   *             dbg::out(tracing) << dbg::prefix() << "Hello!\n";
   *         }
   *         dbg::out(tracing) << dbg::prefix() << "Hello again!\n";
   *         // more stuff
   *     }
   * </pre>
   *
   * You will get the following tracing information:
   *
   * <pre>
   *     *** ->foo (0 in foo.cpp)
   *     ***   ->sub block
   *     ***     Hello!
   *     ***   <-sub block
   *     ***   Hello again!
   *     *** <-foo (0 in foo.cpp)
   * </pre>
   *
   * Don't forget to create named dbg::trace objects. If you create
   * anonymous objects (i.e. you just wrote "dbg::trace(DBG_HERE);")
   * then the destructor will be called immediately, rather than at the
   * end of the block scope, causing invalid trace output.
   *
   * Tracing does not cause assertions to trigger, therefore you will
   * never generate an abort or exception using this object.
   *
   * If you disable the tracing diagnostic @ref level before the trace
   * object's destructor is called you will still get the closing trace
   * output. This is important, otherwise the indentation level of the
   * library would get out of sync. In this case, the closing diagnostic
   * output will have a "note" attached to indicate what has happened.
   *
   * Similarly, if tracing diagnostics are off when the trace object is
   * created, yet subsequencently enabled before the destructor there will
   * be no closing tracing ouput.
   */
  class trace {
   public:

    /**
     * Provide the function name, or some other tracing string.
     *
     * This will not tie the trace object to a particular
     * @ref dbg_source.
     *
     * @param name Tracing block name
     */
    trace(func_name_t name);

    /**
     * @param src  String describing the diagnostic source
     * @param name Tracing block name
     */
    trace(dbg_source src, func_name_t name);

    /**
     * This will not tie the trace object to a particular
     * @ref dbg_source.
     *
     * @param here Supply DBG_HERE
     */
    trace(const source_pos &here);

    /**
     * @param src  String describing the diagnostic source
     * @param here Supply DBG_HERE
     */
    trace(dbg_source src, const source_pos &here);

    ~trace();

   private:

    trace(const trace &);
    trace &operator=(const trace &);

    void trace_begin();
    void trace_end();

    dbg_source        m_src;
    const char       *m_name;
    const source_pos  m_pos;
    bool              m_triggered;
  };

  /**************************************************************************
   * Post conditions
   *************************************************************************/

  /**
   * A post condition class. This utility automates the checking of
   * post conditions using @ref assertion. It requires a member function
   * with the signature:
   * <pre>
   *     bool some_class::invariant() const;
   * </pre>
   *
   * When you create a post_mem_fun object you specify a post condition
   * member function. When the post_mem_fun object is destroyed the
   * postconsition is asserted.
   *
   * This is useful for methods where there are a number of exit points
   * which would make it tedious to put the same @ref dbg::assertion
   * in multiple places.
   *
   * It is also handy when an exception might be thrown and propagated by a
   * funciton, ensuring that a postcondition is first checked. Bear in mind
   * that Bad Things can happen if the @ref assertion_behaviour is
   * assertions_throw and this is triggered via a propagating exception.
   *
   * An example of usage, the do_test method below uses the post_mem_fun
   * object:
   * <pre>
   *     class test
   *     {
   *         public:
   *             test() : a(10) {}
   *             do_test()
   *             {
   *                 dbg::post_mem_fun<test>
   *                    post(dbg::info, this, &test::invariant, DBG_HERE);
   *                 a = 9;
   *                 if (SOME_CONDITION)
   *                 {
   *                     return;                                      // (*)
   *                 }
   *                 else if (SOME_OTHER_CONDITION)
   *                 {
   *                     throw std::exception();                      // (*)
   *                 }
   *                                                                  // (*)
   *             }
   *         private:
   *             bool invariant()
   *             {
   *                 return a == 10;
   *             }
   *             int a;
   *     };
   * </pre>
   * The post condition will be asserted at each point marked (*).
   *
   * @see dbg::post
   */
  template <class obj_t>
  class post_mem_fun {
   public:

    /**
     * The type of the contraint function. It returns a bool and
     * takes no parameters.
     */
    typedef bool (obj_t::*fn_t)();

    /**
     * @param lvl  Diagnostic level
     * @param obj  Object to invoke @p fn on (usually "this")
     * @param fn   Post condition member function
     * @param here Supply DBG_HERE
     */
    post_mem_fun(level lvl, obj_t *obj, fn_t fn, const source_pos &pos)
      : m_lvl(lvl), m_src(0), m_obj(obj), m_fn(fn), m_pos(pos) {}

    /**
     * @param lvl  Diagnostic level
     * @param src  String describing the diagnostic source
     * @param obj  Object to invoke @p fn on (usually "this")
     * @param fn   Post condition member function
     * @param here Supply DBG_HERE
     */
    post_mem_fun(level lvl, dbg_source src,
                 obj_t *obj, fn_t fn, const source_pos &pos)
      : m_lvl(lvl), m_src(src), m_obj(obj), m_fn(fn), m_pos(pos) {}

    /**
     * Overloaded version of constructor which defaults to the
     * @ref warning diagnostic level.
     *
     * @param obj  Object to invoke @p fn on (usually "this")
     * @param fn   Post condition member function
     * @param here Supply DBG_HERE
     */
    post_mem_fun(obj_t *obj, fn_t fn, const source_pos &pos)
      : m_lvl(dbg::warning), m_src(0),
        m_obj(obj), m_fn(fn), m_pos(pos) {}

    /**
     * Overloaded version of constructor which defaults to the
     * @ref warning diagnostic level.
     *
     * @param src  String describing the diagnostic source
     * @param obj  Object to invoke @p fn on (usually "this")
     * @param fn   Post condition member function
     * @param here Supply DBG_HERE
     */
    post_mem_fun(dbg_source src, obj_t *obj, fn_t fn,
                 const source_pos &pos)
      : m_lvl(dbg::warning), m_src(src),
        m_obj(obj), m_fn(fn), m_pos(pos) {}

    /**
     * The destructor asserts the post condition.
     */
    ~post_mem_fun() {
      assertion(m_lvl, m_src,
                assert_info((m_obj->*m_fn)(), "post condition",
                            m_pos.line, m_pos.func, m_pos.file, m_pos.src));
    }

   private:

    const level       m_lvl;
    const dbg_source  m_src;
    obj_t            *m_obj;
    fn_t              m_fn;
    const source_pos  m_pos;
  };

  /**
   * A post condition class. Unlike @ref post_mem_fun, this class
   * calls a non-member function with signature:
   * <pre>
   *     bool some_function();
   * </pre>
   *
   * Otherwise, use it identically to the @ref post_mem_fun.
   *
   * @see dbg::post_mem_fun
   */
  class post {
   public:

    /**
     * The type of the contraint function. It returns a bool and
     * takes no parameters.
     */
    typedef bool (*fn_t)();

    /**
     * @param lvl  Diagnostic level
     * @param fn   Post condition function
     * @param here Supply DBG_HERE
     */
    post(level lvl, fn_t fn, const source_pos &pos)
      : m_lvl(lvl), m_src(0), m_fn(fn), m_pos(pos) {}

    /**
     * @param lvl  Diagnostic level
     * @param src  String describing the diagnostic source
     * @param fn   Post condition function
     * @param here Supply DBG_HERE
     */
    post(level lvl, dbg_source src, fn_t fn, const source_pos &pos)
      : m_lvl(lvl), m_src(src), m_fn(fn), m_pos(pos) {}

    /**
     * Overloaded version of constructor which defaults to the
     * @ref warning diagnostic level.
     *
     * @param fn   Post condition function
     * @param here Supply DBG_HERE
     */
    post(fn_t fn, const source_pos &pos)
      : m_lvl(dbg::warning), m_src(0), m_fn(fn), m_pos(pos) {}

    /**
     * Overloaded version of constructor which defaults to the
     * @ref warning diagnostic level.
     *
     * @param src  String describing the diagnostic source
     * @param fn   Post condition function
     * @param here Supply DBG_HERE
     */
    post(dbg_source src, fn_t fn, const source_pos &pos)
      : m_lvl(dbg::warning), m_src(src), m_fn(fn), m_pos(pos) {}

    /**
     * The destructor asserts the post condition.
     */
    ~post() {
      assertion(m_lvl, m_src,
                assert_info(m_fn(), "post condition",
                            m_pos.line, m_pos.func, m_pos.file, m_pos.src));
    }

   private:

    level            m_lvl;
    const dbg_source m_src;
    fn_t             m_fn;
    const source_pos m_pos;
  };

  /**************************************************************************
   * Compile time assertions
   *************************************************************************/

  /**
   * If we need to assert a constraint that can be calculated at compile
   * time, then it would be advantageous to do so - moving error detection
   * to an earlier phase in development is always a Good Thing.
   *
   * This utility allows you to do this. You use it like this:
   *
   * <pre>
   *     enum { foo = 4, bar = 6 };
   *     compile_assertion<(foo > bar)>();
   * </pre>
   *
   * There is a particular point to observe here. Although the
   * expression is now a template parameter, it is important to contain it
   * in parentheses. This is simply because the expression contains a ">"
   * which otherwise would be taken by the compiler to be the closing of
   * the template parameter. Although not all expressions require this,
   * it is good practice to do it at all times.
   */
  template <bool expression>
  class compile_assertion;
  template <>
  class compile_assertion<true> {};

#else

  /**************************************************************************
   * Non-debug stub versions
   *************************************************************************/

  /*
   * With debugging switched off we generate null versions of the above
   * definitions.
   *
   * Given a good compiler and a strong prevailing headwind, these will
   * optimise away to nothing.
   */

#define DBG_HERE         ((void*)0)
#define DBG_ASSERTION(a) ((void*)0)

  //enum { default_source = 0xdead };
  const dbg_source default_source = 0;

  /**
   * In non-debug versions, this class is used to replace an ostream
   * so that code will compile away. Do not use it directly.
   *
   * @internal
   */
  class null_stream {
   public:
#ifdef _MSC_VER
    null_stream &operator<<(void *)        {
      return *this;
    }
    null_stream &operator<<(const void *)  {
      return *this;
    }
    null_stream &operator<<(long)          {
      return *this;
    }
#else
    template <class otype>
    null_stream &operator<<(const otype &) {
      return *this;
    }
#endif

    template <class otype>
    null_stream &operator<<(otype &)       {
      return *this;
    }
    null_stream &operator<<(std::ostream& (*)(std::ostream&)) {
      return *this;
    }
  };

  struct prefix {
    prefix() {} prefix(level) {}
  };
  struct indent {
    indent() {} indent(level) {}
  };

  inline void        enable(level, bool)                                 {}
  inline void        enable(level, dbg_source, bool)                     {}
  inline void        enable_all(level, bool)                             {}
  inline null_stream out(level, dbg_source)         {
    return null_stream();
  }
  inline null_stream out(level)                     {
    return null_stream();
  }
  inline void        attach_ostream(level, std::ostream &)               {}
  inline void        attach_ostream(level, dbg_source, std::ostream &)   {}
  inline void        detach_ostream(level, std::ostream &)               {}
  inline void        detach_ostream(level, dbg_source, std::ostream &)   {}
  inline void        detach_all_ostreams(level)                          {}
  inline void        detach_all_ostreams(level, dbg_source)              {}
  inline null_stream info_out()                     {
    return null_stream();
  }
  inline null_stream warning_out()                  {
    return null_stream();
  }
  inline null_stream error_out()                    {
    return null_stream();
  }
  inline null_stream fatal_out()                    {
    return null_stream();
  }
  inline null_stream trace_out()                    {
    return null_stream();
  }
  inline void        set_prefix(const char *)                            {}
  inline void        enable_level_prefix(bool)                           {}
  inline void        enable_time_prefix(bool)                            {}

  inline void        set_assertion_behaviour(level, assertion_behaviour) {}
  inline void        set_assertion_period(dbgclock_t)                    {}
  inline void        assertion(level, dbg_source, void *)                {}
  inline void        assertion(level, void *)                            {}
  inline void        assertion(dbg_source, void *)                       {}
  inline void        assertion(void *)                                   {}
  inline void        sentinel(level, dbg_source, void *)                 {}
  inline void        sentinel(level, void *)                             {}
  inline void        sentinel(dbg_source, void *)                        {}
  inline void        sentinel(void *)                                    {}
  inline void        unimplemented(level, dbg_source, void *)            {}
  inline void        unimplemented(level, void *)                        {}
  inline void        unimplemented(dbg_source, void *)                   {}
  inline void        unimplemented(void *)                               {}
  inline void        check_ptr(level, dbg_source, void *, void *)        {}
  inline void        check_ptr(level, void *, void *)                    {}
  inline void        check_ptr(dbg_source, void *, void *)               {}
  inline void        check_ptr(void *, void *)                           {}
  inline void        check_bounds(level, void *, int, int, void *)       {}
  inline void        check_bounds(level, dbg_source, int, void*, void*)  {}
  inline void        check_bounds(level, dbg_source, int, int,
                                  void *, void *)                        {}
  inline void        check_bounds(level, int, void *, void*)             {}
  inline void        check_bounds(void *, int, void *, void *)           {}
  inline void        check_bounds(int, void *, void *)                   {}
  inline   void	       init() {}
  class trace {
   public:
    trace(const char *fn_name)                                     {}
    trace(dbg_source, const char *fn_name)                         {}
    trace(void *here)                                              {}
    trace(dbg_source, void *here)                                  {}
    ~trace()                                                       {}
  };

  template <class obj_t>
  class post_mem_fun {
   public:
    typedef bool (obj_t::*fn_t)();
    post_mem_fun(level, void *, fn_t, void *)                      {}
    post_mem_fun(level, dbg_source, void *, fn_t, void *)          {}
    post_mem_fun(void *, fn_t, void *)                             {}
    post_mem_fun(dbg_source, void *, fn_t, void *)                 {}
    ~post_mem_fun()                                                {}
  };
  class post {
   public:
    typedef bool(*fn_t)();
    post(level, fn_t, void *)                                      {}
    post(level, dbg_source, fn_t, void *)                          {}
    post(fn_t, void *)                                             {}
    post(dbg_source, fn_t, void *)                                 {}
    ~post()                                                        {}
  };

  template <bool expression>
  class compile_assertion {};

#endif
}

#endif
