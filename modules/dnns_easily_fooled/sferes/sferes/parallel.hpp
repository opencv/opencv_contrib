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




#ifndef PARALLEL_HPP_
#define PARALLEL_HPP_

#ifndef NO_PARALLEL
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/partitioner.h>
#include <tbb/parallel_sort.h>
#endif

// parallel for can be deactivated by defining NO_PARALLEL
// maximum of threads can be specified by defining NB_THREADS
namespace sferes {
  namespace parallel {

#ifndef NO_PARALLEL
    typedef tbb::blocked_range<size_t> range_t;

#ifdef NB_THREADS
    static void init() {
      static tbb::task_scheduler_init init(NB_THREADS);
    }
#else
    static void init() {
      static tbb::task_scheduler_init init;
    }
#endif

    template<typename Range, typename Body>
    inline void p_for(const Range& range, const Body& body) {
      tbb::parallel_for(range, body);
    }

    template<typename Range, typename Body>
    inline void p_for(const Range& range, Body& body) {
      tbb::parallel_for(range, body);
    }


    template<typename T1, typename T2, typename T3>
    void sort(T1 i1, T2 i2, T3 comp) {
      tbb::parallel_sort(i1, i2, comp);
    }
#else
    class PRange {
     public:
      PRange(size_t b, size_t e) : _begin(b), _end(e) {}
      PRange(const PRange& o) : _begin(o._begin), _end(o._end) {}
      size_t begin() const {
        return _begin;
      }
      size_t end() const   {
        return _end;
      }
     protected:
      size_t _begin, _end;
    };
    typedef PRange range_t;

    static void init() {}

    template<typename Range, typename Body>
    inline void p_for(const Range& range, const Body& body) {
      body(range);
    }
    // non const version
    template<typename Range, typename Body>
    inline void p_for(const Range& range, Body& body) {
      body(range);
    }

    template<typename T1, typename T2, typename T3>
    void sort(T1 i1, T2 i2, T3 comp) {
      std::sort(i1, i2, comp);
    }
#endif
  };

}

#endif
