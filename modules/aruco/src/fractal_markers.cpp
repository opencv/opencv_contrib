#include "precomp.hpp"
#include "opencv2/aruco/fractal_markers.hpp"

namespace cv {
namespace aruco {

class FractalDictionaryImpl : public FractalDictionary {};

Ptr<FractalDictionary> getFractalDictionary(int dictType) {
    return makePtr<FractalDictionaryImpl>();
}

} // namespace aruco
} // namespace cv