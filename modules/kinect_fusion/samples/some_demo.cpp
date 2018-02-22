//TODO: license here

//TODO: other includes
#include <opencv2/phase_unwrapping.hpp>

using namespace cv;
using namespace std;

//TODO: keys
static const char* keys =
{
    "{@arg1 | | Path of the warrior }"
};

//TODO: help
static void help()
{
    cout << "\nThis example shows something"
            "\nI dont know what \n"
         << endl;
}
int main(int argc, char **argv)
{
    phase_unwrapping::HistogramPhaseUnwrapping::Params params;

//TODO: parse args
    CommandLineParser parser(argc, argv, keys);
    String param1 = parser.get<String>(0);

//TODO: code here
    return 0;
}
