// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "precomp.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <numeric>
#include <fstream>



using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::saliency;

namespace cv
{
namespace saliency
{

DeepGaze1::DeepGaze1( string net_proto, string net_caffemodel )
{
    net = dnn::readNetFromCaffe( net_proto, net_caffemodel );
    layers_names.push_back("conv5");
    double tmp[] = {0.471836,0.459718,0.457611,0.480085,0.48823,0.462776,0.483445,0.461641,0.483469,0.47846,0.470172,0.383401,0.459975,0.478768,0.576305,0.469454,0.480448,0.435255,0.493571,0.48153,0.441619,0.45436,0.451791,0.476142,0.475955,0.468466,0.479836,0.530578,0.481301,0.454639,0.464725,0.45218,0.502245,0.44709,0.483469,0.500928,0.469562,0.46483,0.453324,0.461538,0.475741,0.480432,0.485595,0.462417,0.495092,0.471557,0.495046,0.459551,0.43668,0.505385,0.478419,0.492535,0.303292,0.475142,0.459992,0.454734,0.466868,0.450649,0.479587,0.434151,0.471309,0.460742,0.49318,0.524707,0.470968,0.478263,0.469935,0.459639,0.490684,0.465349,0.44842,0.481436,0.488862,0.468849,0.492233,0.467677,0.448416,0.474485,0.47684,0.492617,0.455164,0.46794,0.463009,0.47758,0.46629,0.495621,0.464325,0.473217,0.459664,0.478029,0.438637,0.447406,0.438148,0.455966,0.473499,0.473359,0.466213,0.525776,0.434224,0.464641,0.475869,0.501644,0.485892,0.483617,0.47226,0.482615,0.448091,0.460951,0.470457,0.469719,0.474948,0.516341,0.474467,0.429576,0.460061,0.446831,0.429813,0.479859,0.509008,0.504804,0.477351,0.461487,0.445481,0.44935,0.482019,0.469048,0.473205,0.460742,0.474685,0.461985,0.497119,0.464336,0.469783,0.464748,0.477133,0.484101,0.491574,0.591169,0.47327,0.467959,0.479773,0.465179,0.456533,0.42534,0.457655,0.474379,0.482501,0.491678,0.558077,0.473311,0.483722,0.474757,0.46874,0.459033,0.483051,0.475974,0.449861,0.456586,0.462686,0.46992,0.424458,0.492504,0.450006,0.468069,0.450585,0.442672,0.460277,0.460656,0.449303,0.470552,0.433665,0.47603,0.449626,0.471062,0.481555,0.427269,0.424295,0.588326,0.475818,0.484487,0.496265,0.480074,0.45834,0.469174,0.474869,0.49295,0.458737,0.461799,0.487588,0.488148,0.47734,0.480953,0.478616,0.470873,0.456516,0.461151,0.497269,0.449723,0.414189,0.473214,0.47472,0.478068,0.454312,0.485553,0.43564,0.469596,0.450846,0.488699,0.481056,0.419303,0.479696,0.471458,0.456179,0.465579,0.449656,0.459427,0.475431,0.518732,0.45971,0.51276,0.475805,0.467066,0.455423,0.462425,0.468577,0.429871,0.467098,0.467196,0.48245,0.496047,0.439613,0.446267,0.478326,0.463222,0.466251,0.475164,0.460792,0.407577,0.475157,0.465814,0.480478,0.490252,0.485834,0.455555,0.488025,0.472621,0.482393,0.48254,0.500558,0.466278,0.478975,0.423606,0.482795,0.486593,0.488191,0.483121,5.96006};
    weights = vector<double>( tmp, tmp + 257 );
}

DeepGaze1::DeepGaze1( string net_proto, string net_caffemodel, vector<string> selected_layers, unsigned n_weights )
{
    net = dnn::readNetFromCaffe( net_proto, net_caffemodel );
    layers_names = selected_layers;
    weights = vector<double>( n_weights, 0 );
    for ( unsigned i = 0; i < weights.size(); i++ )
    {
        weights[i] = (double)( rand() % 10 ) / 10.0;
    }
}

DeepGaze1::DeepGaze1( string net_proto, string net_caffemodel, vector<string> selected_layers, vector<double> i_weights )
{
    net = dnn::readNetFromCaffe( net_proto, net_caffemodel );
    layers_names = selected_layers;
    weights = i_weights;
}

DeepGaze1::~DeepGaze1(){}

vector<Mat> DeepGaze1::featureMapGenerator( Mat img, Size input_size )
{
    vector<Mat> featureMaps;
    Mat me(input_size, CV_64F, Scalar::all(0.0));
    Mat std(input_size, CV_64F, Scalar::all(0.0));
    Mat temp(input_size, CV_64F, Scalar::all(0.0));


    resize( img, img, input_size, 0, 0, INTER_AREA );//hard coded
    Mat inputBlob = blobFromImage( img );   //Convert Mat to batch of images
    net.setInput( inputBlob, "data" );
    //net.forward();
    //net.setBlob(".data", inputBlob);        //set the network input
    //net.forward();                          //compute output
    for ( unsigned i = 0; i < layers_names.size(); i++ )
    {
        //Mat blob_set = net.getBlob(layers_names[i]);
        Mat blob_set = net.forward( layers_names[i] );
        for ( int j = 0; j < blob_set.size[1]; j++ )
        {
            Mat m, s, blob = Mat( blob_set.size[2], blob_set.size[3], CV_64F, blob_set.ptr(0, j) ).clone();
            resize( blob, blob, input_size, 0, 0, INTER_CUBIC );
            featureMaps.push_back( blob );
        }
    }
    for ( unsigned i = 0; i < featureMaps.size(); i++ )
    {
        me += ( featureMaps[i] / (double)featureMaps.size() );
    }
    for ( unsigned i = 0; i < featureMaps.size(); i++ )
    {
        pow( featureMaps[i] - me, 2, temp );
        std += (temp / (double)featureMaps.size());
    }
    pow( std, 0.5, std );
    for ( unsigned i = 0; i < featureMaps.size(); i++ )
    {
        featureMaps[i] -= me;
        divide( featureMaps[i], std, featureMaps[i] );
    }
    Mat centerBias = ( Mat_<double>(15, 15) << 0.136605132339619,0.254911885194451,0.354975848201554,0.432118422919642,0.492788960570534,0.537169372900984,0.559893362856313,0.570408994503682,0.558327108322080,0.529592014101891,0.483151700229565,0.421134751334104,0.339516463333039,0.240333898610746,0.119040524151926,0.255057036816424,0.373463164527964,0.471695245848319,0.548331995771048,0.610621682003216,0.652904303445073,0.678084927709815,0.686289988540627,0.675100404076781,0.648787013529810,0.601363163606808,0.539724655027488,0.457897183544702,0.359051684872921,0.237121060715778,0.354996162889476,0.471574664262948,0.569824563418442,0.647926075225057,0.710453524703313,0.751807537518647,0.777982678740224,0.785604558552769,0.773105643206983,0.747143093637550,0.700960041197720,0.637036644172802,0.555488832006236,0.456278688906414,0.336404833368141,0.432114886533253,0.548519754764192,0.647919735424171,0.726252091271344,0.787633820027685,0.830453686804553,0.855159388723024,0.864261146191168,0.851130919200443,0.823782075450280,0.777941301454146,0.714197202377273,0.632042767206447,0.533375632722201,0.413459898633110,0.492846224526104,0.610458063998438,0.710273041291813,0.787934734560166,0.848818377335713,0.892669644305876,0.915962511030596,0.925983774442787,0.914078884852605,0.886340426175551,0.838921069614458,0.776522338759063,0.694910735314247,0.595717337417090,0.474404562107594,0.537258377108661,0.652654274132924,0.751668431422994,0.830557252057301,0.892264877678679,0.934387776714834,0.960294475640559,0.968153776825753,0.957118891968944,0.930719438034762,0.882668936995775,0.818004698514852,0.736335421611659,0.637250235468980,0.518321616011027,0.559912410908083,0.678060440904210,0.778035916610351,0.855337573494623,0.916090961317328,0.960268532575822,0.983151329506359,0.993265865328518,0.981393158586731,0.953452714649956,0.906806535966130,0.844266579567292,0.762471436321223,0.663436280862256,0.542292468493236,0.570471859736804,0.686367557103361,0.785571470982568,0.863984557875191,0.925787582513478,0.968184312606637,0.993160551514547,1.00071969560062,0.990820131015439,0.963788669203278,0.916062100871062,0.851941275823728,0.770061877665113,0.671049079061439,0.551460475500773,0.558354004396870,0.675218828625687,0.773141022132423,0.851023228938218,0.914111180726610,0.957097877210958,0.981435236380489,0.990830375805955,0.979069435484073,0.951285941586670,0.904001536547754,0.841050003441305,0.759279958742245,0.660299547838899,0.539966072338975,0.529794530767286,0.648426268685882,0.747085236556856,0.823784188854240,0.886061506412046,0.930850736147926,0.953129788839557,0.963752221613668,0.951512839210700,0.924189623933934,0.877773421560175,0.814688659956347,0.733050627339970,0.633967523008819,0.512144357629488,0.483094193799984,0.601317535642541,0.700936072455337,0.778084086878178,0.839052826457435,0.882842688790194,0.906345815545706,0.916276056981117,0.904027650739533,0.877827433807473,0.831763094963022,0.767578814024913,0.685653480474771,0.586759596897176,0.465090771371211,0.421344630481700,0.539705161668206,0.637059989497880,0.714213691485248,0.776688317470029,0.818187784047592,0.844251497483870,0.851844570961231,0.841044566248117,0.814778195034621,0.767546270221294,0.705699445037065,0.623983961385159,0.525082406012993,0.403548799412053,0.339352953813616,0.458020160692771,0.555130309401582,0.632633247554868,0.694581915649390,0.736338072460011,0.762425604772228,0.770108192322197,0.759369952472952,0.732901801094476,0.685633947666787,0.623974569677122,0.542296497992531,0.443313473703102,0.321653790809172,0.240452590442239,0.358845767555164,0.456323501276454,0.533311273531990,0.595861650031113,0.637378281509690,0.663434282045727,0.670883353894946,0.660356369332519,0.633940197197224,0.586949913320820,0.525019901253340,0.443239391952182,0.341581133756428,0.221496442250004,0.118883254021697,0.237092401238019,0.336442576365934,0.413042016689593,0.474758090922195,0.518099556977193,0.541973522382581,0.551465525482662,0.539575142928788,0.512205675560351,0.465197135656097,0.403297133574821,0.321504846482331,0.221359680451730,0.100335071938095);
    resize( centerBias, centerBias, input_size, 0, 0, INTER_CUBIC );
    featureMaps.push_back( centerBias );
    return featureMaps;
}

bool DeepGaze1::computeSaliencyImpl( InputArray image, OutputArray saliencyMap )
{
    CV_Assert( !(image.getMat().empty()) );
    vector<Mat> featureMaps = featureMapGenerator( image.getMat(), Size(227, 227) );
    saliencyMap.assign( softmax( comb( featureMaps, weights ) ) );
    for ( unsigned i = 0; i < weights.size(); i++ ) cout << weights[i] << ",";
    cout << endl;
    return true;
}

Mat DeepGaze1::saliencyMapGenerator( Mat input_image, Size input_size )
{
//raw saliency map generate
    CV_Assert( !(input_image.empty()) );
    vector<Mat> featureMaps = featureMapGenerator( input_image, input_size );
    return softmax( comb( featureMaps, weights ) );
}

Mat DeepGaze1::comb( vector<Mat>& featureMaps, vector<double> wei )
{
    Mat res( featureMaps[0].rows, featureMaps[0].cols, CV_64F, Scalar::all(0.0) );

    for ( unsigned i = 0; i < featureMaps.size(); i++ )
    {
        Mat temp = featureMaps[i].clone();
        temp *= wei[i];
        res += temp;
    }
    GaussianBlur( res, res, Size(35, 35), 0, 0, BORDER_CONSTANT );
    return res;
}

Mat DeepGaze1::softmax( Mat res )
{
    exp(res, res);
    double temp = sum(res).val[0];
    res = res / temp;

    return res;
}

vector<unsigned> DeepGaze1::batchIndex( unsigned total, unsigned batchSize )
{
    srand(0);
    vector<unsigned> allIndex(total, 0);

    for ( unsigned i = 0; i < total; i++ )
    {
        allIndex[i] = i;
    }
    for ( int i = batchSize - 1; i >= 0; i-- )
    {
        swap(allIndex[i], allIndex[rand() % total]);
    }
    return vector<unsigned>( allIndex.begin(), allIndex.begin() + batchSize );
}



vector<unsigned> DeepGaze1::fixationLoc( Mat img, Size input_size )
{
    CV_Assert( !(img.empty()) );
    vector<unsigned> randIndex;
    resize( img, img, input_size, 0, 0, INTER_AREA );
    vector<unsigned> fixation;
    vector<pair<unsigned, unsigned> > match;
    fixation.assign(img.datastart, img.dataend);
    for ( unsigned i = 0; i < fixation.size(); i++ )
    {
        match.push_back( pair<unsigned, unsigned>(fixation[i], i));
    }
    sort( match.begin(), match.end(), greater<pair<unsigned, unsigned> >() );
    //sort(match.begin(), match.end(), [](pair<unsigned, unsigned> a, pair<unsigned, unsigned> b) {
    //    return b.first < a.first;
    //});
    for ( unsigned i = 0 ; ((i < match.size()) && (match[i].first > 0)) ; i++ )
    {
        randIndex.push_back( match[i].second );
    }
    img.release();
    return randIndex;
}

double DeepGaze1::loss( vector<double> saliency_sample, vector<double> wei )
{
    double res = 0, l1 = 0, l2 = 0;

    for ( unsigned i = 0;i < wei.size();i++ )
    {
        l1 += abs(wei[i]);
    }
    for ( unsigned i = 0;i < wei.size();i++ )
    {
        l2 += wei[i] * wei[i];
    }
    for ( unsigned i = 0; i < saliency_sample.size(); i++ )
    {
        res -= log( saliency_sample[i] ) / saliency_sample.size();
    }
    return res + 0.001 * l1 / sqrt(l2);
}

vector<double> DeepGaze1::mapSampler( Mat saliency, vector<unsigned> randIndex )
{
    vector<double> saliency_sample;
    for ( unsigned i = 0 ; i < randIndex.size() ; i++ )
    {
        saliency_sample.push_back( saliency.at<double>( randIndex[i] / saliency.size[1], randIndex[i] % saliency.size[1] ) );
    }
    return saliency_sample;
}

vector<double> DeepGaze1::evalGrad( vector<Mat>& featureMaps, vector<unsigned>& randIndex, vector<double> wei, Size input_size )
{
    vector<double> grad( featureMaps.size(), 0 );

    Mat c = comb( featureMaps, wei );
    Mat saliency_old = c.clone();
    saliency_old = softmax( saliency_old );
    vector<double> tt_old = mapSampler( saliency_old, randIndex );
    double loss_old = 0;
    loss_old = loss( tt_old, wei );
    for ( unsigned i = 0; i < wei.size(); i++ )
    {
        Mat saliency_new = c.clone();
        Mat temp( input_size, CV_64F, Scalar::all(0.0) );
        temp += 0.0000000001 * featureMaps[i];
        GaussianBlur( temp, temp, Size(35, 35), 0, 0, BORDER_CONSTANT );
        saliency_new += temp;
        saliency_new = softmax( saliency_new );
        vector<double> weights_new( wei );
        weights_new[i] += 0.0000000001;
        double loss_new = 0;
        vector<double> tt = mapSampler( saliency_new, randIndex );
        loss_new = loss( tt, weights_new );
        grad[i] = ( loss_new - loss_old ) / 0.0000000001;
    }
    return grad;
}

void DeepGaze1::training( vector<Mat>& images, vector<Mat>& fixMaps, int iteration, unsigned batch_size, double momentum, double alpha, double decay, Size input_size )
{
    vector<unsigned> randIndex = batchIndex( (unsigned)images.size(), min( batch_size, (unsigned)images.size() ) );
    vector<unsigned> fixLoc;
    vector<Mat> featureMaps;
    vector<double> grad;
    vector<double> vel(weights.size(), 0);

    unsigned n = 0;
    while ( iteration > 0 )
    {
        for ( unsigned i = 0; i < randIndex.size(); i++ )
        {
            featureMaps = featureMapGenerator( images[randIndex[i]], input_size );
            fixLoc = fixationLoc( fixMaps[randIndex[i]], input_size );
            grad = evalGrad( featureMaps, fixLoc, weights, input_size );
            for ( unsigned j = 0; j < grad.size(); j++ )
            {
                vel[j] = momentum * vel[j] + grad[j];
                weights[j] -= alpha * vel[j] * exp(-decay * n);
            }
            n++;
            double avgGrad = accumulate( grad.begin(), grad.end(), 0.0 ) / weights.size();
            double avgWeight = accumulate( weights.begin(), weights.end(), 0.0 ) / weights.size();
            cout << n << " " << avgGrad << " " << avgWeight << endl;
        }
        iteration--;
    }
}

double DeepGaze1::computeAUC( InputArray _saliencyMap, InputArray _fixtionMap )
{
    Mat saliency = _saliencyMap.getMat().clone();
    Mat fixtion = _fixtionMap.getMat().clone();

    if ( saliency.empty() || fixtion.empty() || saliency.dims > 2 || fixtion.dims > 2 )
    {
        cout << "saliency map and fixtion map must be 1 channel and have same size" << endl;
        CV_Assert( !( saliency.empty() || fixtion.empty() || saliency.dims > 2 || fixtion.dims > 2 ) );
        return -1;
    }

    resize ( fixtion, fixtion, Size(saliency.cols, saliency.rows), 0, 0, INTER_CUBIC );
    double mi = 0, ma = 0;
    minMaxLoc( saliency, &mi, &ma );
    saliency -= mi;
    saliency /= ( ma - mi );
    vector<double> threshold_list;
    for ( int i = 0; i < saliency.rows; i++ )
    {
        for ( int j = 0; j < saliency.cols; j++ )
        {
            if ( fixtion.at<uchar>(i, j) > 0 ) threshold_list.push_back( saliency.at<double>(i, j) );
        }
    }
    sort( threshold_list.begin(), threshold_list.end(), greater<double>() );

    vector<double> tp(1, 0), fp(1, 0);
    for ( unsigned i = 0; i < threshold_list.size(); i++ )
    {
        unsigned aboveth = 0;
        for ( int m = 0; m < saliency.rows; m++ )
        {
            for ( int n = 0; n < saliency.cols; n++ )
            {
                if ( saliency.at<double>(m, n) >= threshold_list[i] ) aboveth++;
            }
        }
        tp.push_back( (i + 1.0) / (double)threshold_list.size() );
        fp.push_back( (aboveth - i - 1.0) / (double)(saliency.rows * saliency.cols - threshold_list.size()) );
    }
    tp.push_back(1);
    fp.push_back(1);
    double auc = 0;
    for ( unsigned i = 1; i < tp.size(); i++ )
    {
        auc += (tp[i - 1] + tp[i]) * (fp[i] - fp[i - 1]) / 2;
    }

    return auc;
}

void DeepGaze1::saliencyMapVisualize( InputArray _saliencyMap )
{
    Mat saliency = _saliencyMap.getMat().clone();
    double mi = 0, ma = 0;
    minMaxLoc( saliency, &mi, &ma );
    saliency -= mi;
    saliency /= ( ma - mi );
    imshow( "saliencyVisual", saliency );
    waitKey( 0 );
}
}
}
