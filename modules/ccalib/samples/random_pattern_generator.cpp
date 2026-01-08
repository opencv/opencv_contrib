#include "opencv2/ccalib/randpattern.hpp"

using namespace cv;

const char * usage =
    "\n example command line for generating a random pattern. \n"
    "   random_patterng_generator -iw 600 -ih 850 pattern.png\n"
    "\n";

static void help()
{
    printf("\n This is a sample for generating a random pattern that can be used for calibration.\n"
        "Usage: random_patterng_generator\n"
        "    -iw <image_width> # the width of pattern image\n"
        "    -ih <image_height> # the height of pattern image\n"
        "    filename # the filename for pattern image \n"
        );
    printf("\n %s", usage);
}

int main(int argc, char** argv)
{
    const char* filename = 0;
    Mat pattern;
    int width = 0, height = 0;
    if(argc < 2)
    {
        help();
        return 1;
    }

    for (int i = 1; i < argc; ++i)
    {
        const char* s = argv[i];
        if(strcmp(s, "-iw") == 0)
        {
            if(sscanf(argv[++i], "%d", &width) != 1 || width <= 0)
                return fprintf( stderr, "Invalid pattern image width\n"), -1;
        }
        else if(strcmp(s, "-ih") == 0)
        {
            if(sscanf(argv[++i], "%d", &height) != 1 || height <= 0)
                return fprintf( stderr, "Invalid pattern image height\n"), -1;
        }
        else if( s[0] != '-')
        {
            filename = s;
        }
        else
        {
            return fprintf( stderr, "Unknown option %s\n", s ), -1;
        }
    }

    randpattern::RandomPatternGenerator generator(width, height);
    generator.generatePattern();
    pattern = generator.getPattern();
    imwrite(filename, pattern);
}