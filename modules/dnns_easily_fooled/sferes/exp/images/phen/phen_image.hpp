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


#ifndef PHEN_IMAGE_HPP
#define PHEN_IMAGE_HPP

#include <map>
#include <sferes/phen/indiv.hpp>
#include <modules/nn2/nn.hpp>

#include <modules/nn2/params.hpp>
#include <modules/nn2/gen_hyper_nn.hpp>


// New stuff added ------------------------------------------

#include <cmath>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/join.hpp>
#include <string>

#include "cvmat_serialization.h" // Serialize cv::Mat
#include <glog/logging.h>	// Google Logging

//#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/uuid/uuid_serialize.hpp>	// serialization

// New stuff added ------------------------------------------

namespace sferes
{
  namespace phen
  {
     // hyperneat-inspired phenotype, based on a cppn
    SFERES_INDIV(Image, Indiv)
    {
    public:
    	  Image(): _created_gen(0)
        {
        	boost::uuids::uuid uuid = boost::uuids::random_generator()();
        	_id = uuid;
        }

        /*
         * Get the ID of this organism.
         */
        boost::uuids::uuid id()
        {
        	return _id;
        }

        /*
         * Set the generation when this organism is created.
         */
        void set_created_gen(const size_t generation)
        {
        	_created_gen = generation;
        }

        /*
         * Get the generation when this organism is created.
         */
        size_t created_gen() const
        {
        	return _created_gen;
        }


        template<class Archive>
				void serialize(Archive & ar, const unsigned int version)
        {
					sferes::phen::Indiv<Gen, Fit, Params,  typename stc::FindExact<Image<Gen, Fit, Params, Exact>, Exact>::ret>::serialize(ar, version);
					ar & boost::serialization::make_nvp("uuid", _id.data);
					ar & BOOST_SERIALIZATION_NVP(_created_gen);
        }

      protected:
        boost::uuids::uuid _id;		// The unique id of this organism
        size_t _created_gen;							// The generation when this image is created
    };
  }
}


#endif
