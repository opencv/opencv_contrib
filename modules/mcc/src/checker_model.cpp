#include "opencv2/mcc/checker_model.hpp"
#include <locale>
#include <iostream>
using namespace std;

namespace cv{
namespace mcc{
const float CChartClassicModel::chart[24][9] = {

	//       sRGB              CIE L*a*b*             Munsell Notation
	// ---------------  ------------------------     Hue Value / Chroma
	// R     G      B        L*      a*       b*
	{115.0,  82.0,  68.0,  37.986,  13.555,  14.059,   3.00,  3.70,   3.2},		//1.  dark shin
	{194.0, 150.0, 130.0,  65.711,  18.130,  17.810,   2.20,  6.47,   4.1},		//2.  light skin
	{ 98.0, 122.0, 157.0,  49.927,  -4.880, -21.925,   4.30,  4.95,   5.0},		//3.  blue skin
	{ 87.0, 108.0,  67.0,  43.139, -13.095,  21.905,   6.70,  4.20,   4.1 },	//4.  foliage
	{133.0, 128.0, 177.0,  55.112,   8.844, -25.399,   9.70,  5.47,   6.7},		//5.  blue flower
	{103.0, 189.0, 170.0,  70.719, -33.395,  -0.199,   2.50,  7.00,   6.0},		//6.  bluish green
	{214.0, 126.0,  44.0,  62.661,  36.067,  57.096,   5.00,  6.00,  11.0},		//7.  orange
	{ 80.0,  91.0, 166.0,  40.020,  10.410, -45.964,   7.50,  4.00,  10.7},		//8.  purplish blue
	{193.0,  90.0,  99.0,  51.124,  48.239,  16.248,   2.50,  5.00,  10.0},		//9.  moderate red
	{ 94.0,  60.0, 108.0,  30.325,  22.976, -21.587,   5.00,  3.00,   7.0},		//10. purple
	{157.0, 188.0,  64.0,  72.532, -23.709,  57.255,   5.00,  7.10,   9.1},		//11. yelow green
	{224.0, 163.0,  46.0,  71.941,  19.363,  67.857,  10.00,  7.00,  10.5},		//12. orange yellow
	{ 56.0,  61.0, 150.0,  28.778,  14.179, -50.297,   7.50,  2.90,  12.7},		//13. blue
	{ 70.0, 148.0,  73.0,  55.261, -38.342,  31.370,   0.25,  5.40,  8.65},		//14. green
	{175.0,  54.0,  60.0,  42.101,  53.378,  28.190,   5.00,  4.00,  12.0},		//15. red
	{231.0, 199.0,  31.0,  81.733,   4.039,  79.819,   5.00,  8.00,  11.1},		//16. yellow
	{187.0,  86.0, 149.0,  51.935,  49.986, -14.574,   2.50,  5.00,  12.0},		//17. magenta
	{  8.0, 133.0, 161.0,  51.038, -28.631, -28.638,   5.00,  5.00,   8.0},		//18. cyan
	{243.0, 243.0, 242.0,  96.539,  -0.425,   1.186,   0.00,  9.50,   0.0},		//19. white(.05*)
	{200.0, 200.0, 200.0,  81.257,  -0.638,  -0.335,   0.00,  8.00,   0.0},		//20. neutral 8(.23*)
	{150.0, 160.0, 160.0,  66.766,  -0.734,  -0.504,   0.00,  6.50,   0.0},		//21. neutral 6.5(.44*)
	{122.0, 122.0, 121.0,  50.867,  -0.153,  -0.270,   0.00,  5.00,   0.0},		//22. neutral 5(.70*)
	{ 58.0,  85.0,  85.0,  35.656,  -0.421,  -1.231,   0.00,  3.50,   0.0},		//23. neutral 3.5(.1.05*)
	{ 52.0,  52.0,  52.0,  20.461,  -0.079,  -0.973,   0.00,  2.00,   0.0}		//24. black(1.50*)

};



CChartClassicModel::CChartClassicModel()
{

	// Cie L*a*b* values use illuminant D50 2 degree observer sRGB values for
	// for iluminante D65.

	size = cv::Size2i(4,6);

	// model chart box
	// -------------------- -
	// | -- - -- -
	// | | 1 | | 2 | ... 6x4(11.0x8.25)
	// | -- - -- -
	// | -- -
	// | | 7 |
	// | -- -
	// |
	//

	boxsize = cv::Size2f(11.25, 16.75);
	box.resize(4);
	box[0] = cv::Point2f( 0.00,  0.00);
	box[1] = cv::Point2f(16.75,  0.00);
	box[2] = cv::Point2f(16.75, 11.25);
	box[3] = cv::Point2f( 0.00, 11.25);

	// model chart
	// -------------------- -
	// | .---------.
	// | |			|
	// | |   .		| 2.5x2.5
	// | |	 1.25	|
	// | |			|
	// | .---------.
	// |
	//

	float cellx  = 2.5, celly = 2.5;
	float step = 0.25;
	float dcenter = 1.25;
	int n = 4, m = 6;

	center.resize(24);
	cellchart.resize(24 * 4);
	float x, y;
	int k = 0;

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			x = j*(cellx + step) + step;
			y = i*(celly + step) + step;

			// cell
			cellchart[4 * k + 0] = cv::Point2f(x, y);
			cellchart[4 * k + 1] = cv::Point2f(x + cellx, y);
			cellchart[4 * k + 2] = cv::Point2f(x + cellx, y + celly);
			cellchart[4 * k + 3] = cv::Point2f(x, y + celly);

			// center
			center[k] = cv::Point2f(x + dcenter, y + dcenter);
			k++;
		}
	}
}


CChartClassicModel::~CChartClassicModel()
{
}

bool CChartClassicModel::
evaluate(const SUBCCMModel & subModel, int & offset, int & iTheta, float & error)
{

	float tError;
	int tTheta, tOffset;
	error = INFINITY;
	bool beval = false;

	// para todas las orientaciones
	// min_{ theta,dt } | CC_e - CC |
	for (tTheta = 0; tTheta < 4; tTheta++)
	{
		if (match(subModel, tTheta, tError, tOffset) && tError < error) {
			error = tError; iTheta = tTheta; offset = tOffset;
			beval = true;
		}
	}

	return beval;

}

void CChartClassicModel::
rotate90() {

	size = cv::Size2i(size.height, size.width);

	float cellx = 2.5, celly = 2.5;
	float step = 0.25;
	float dcenter = 1.25;
	int n = size.width, m = size.height;
	float x, y;
	int k = 0;

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			x = j*(cellx + step) + step;
			y = i*(celly + step) + step;

			// cell
			cellchart[4 * k + 0] = cv::Point2f(y, x);
			cellchart[4 * k + 1] = cv::Point2f(y + cellx, x);
			cellchart[4 * k + 2] = cv::Point2f(y + cellx, x + celly);
			cellchart[4 * k + 3] = cv::Point2f(y, x + celly);

			// center
			center[k] = cv::Point2f(y + dcenter, x + dcenter);
			k++;
		}

	}

	boxsize = cv::Size2f(boxsize.height, boxsize.width);
	box[0] = cv::Point2f(0.00, 0.00);
	box[1] = cv::Point2f(boxsize.width, 0.00);
	box[2] = cv::Point2f(boxsize.width, boxsize.height);
	box[3] = cv::Point2f(0.00, boxsize.height);


}

void CChartClassicModel::
copyToColorMat(cv::Mat & lab, int cs)
{
	int N, M, k;

	N = size.width; M = size.height;
	cv::Mat im_lab_org(N, M, CV_32FC3);
	int type_color = 3 * cs;
	k = 0;

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < M; j++)
		{
			cv::Vec3f &lab = im_lab_org.at<cv::Vec3f>(i, j);
			lab[0] = chart[k][type_color + 0];
			lab[1] = chart[k][type_color + 1];
			lab[2] = chart[k][type_color + 2];
			k++;
		}
	}

	lab = im_lab_org;

}


bool CChartClassicModel::
match(const SUBCCMModel & subModel, int iTheta, float & error, int & ierror)
{

	int N, M, k;

	N = size.width; M = size.height;
	cv::Mat im_lab_org(N, M, CV_32FC3);
	int type_color = 3;
	k = 0;

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < M; j++)
		{
			cv::Vec3f &lab = im_lab_org.at<cv::Vec3f>(i, j);
			lab[0] = chart[k][type_color + 0];
			lab[1] = chart[k][type_color + 1];
			lab[2] = chart[k][type_color + 2];
			k++;
		}
	}

	rot90(im_lab_org, iTheta);
	N = im_lab_org.rows; M = im_lab_org.cols;

	int n, m;
	n = subModel.color_size.height;
	m = subModel.color_size.width;

	// boundary condition
	if (N < n || M < m) return false;


	// rgb to La*b*
	cv::Mat rgb_est = subModel.sub_chart;
	cv::Mat lab_est;

	// RGB color space
	//cv::cvtColor(rgb_est, lab_est, COLOR_BGR2RGB);

	// Lab color space
	//rgb_est *= 1/255;
	cv::cvtColor(rgb_est, lab_est, COLOR_BGR2Lab);


	int nN, mM;
	nN = N - n + 1; mM = M - m + 1;
	std::vector<float> lEcm(nN*mM);
	k = 0;
	for (size_t i = 0; i < nN; i++)
	{
		for (size_t j = 0; j < mM; j++)
		{
			cv::Mat lab_curr, lab_roi;
			lab_roi = im_lab_org(cv::Rect(j, i, m, n));
			lab_roi.copyTo(lab_curr);
			lab_curr = lab_curr.t();
			lab_curr = lab_curr.reshape(3, n*m);

			// Mean squared error
			// ECM = 1 / N sum_i(Y - Yp) ^ 2
			lEcm[k] = dist_color_lab(lab_curr, lab_est) / (M*N);
			k++;
		}
	}


	// minimo
	error = lEcm[0]; ierror = 0;
	for (size_t i = 1; i < lEcm.size(); i++)
		if (error > lEcm[i]) { error = lEcm[i]; ierror = i; }

	return true;

}



float CChartClassicModel::
dist_color_lab(const cv::Mat & lab1, const cv::Mat & lab2)
{

	int N = lab1.rows;
	float dist = 0, dist_i;

	for (size_t i = 0; i < N; i++)
	{
		cv::Vec3f v1 = lab1.at<cv::Vec3f>(i, 0);
		cv::Vec3f v2 = lab2.at<cv::Vec3f>(i, 0);
		v1[0] = 1; v2[0] = 1; // L <- 0

		// eculidian
		cv::Vec3f v = v1 - v2;
		dist_i = v.dot(v);
		dist +=  sqrt(dist_i);

		// coseno
		//float costh = v1.dot(v2) / (norm(v1)*norm(v2));
		//dist += 1 - (1 + costh) / 2;


	}

	dist /= N;
	return dist;

}

void CChartClassicModel::
rot90(cv::Mat & mat, int itheta) {

	//1=CW, 2=CCW, 3=180
	switch (itheta) {
	case 1://transpose+flip(1)=CW
		transpose(mat, mat);
		flip(mat, mat, 1);
		break;
	case 2://flip(-1)=180
		flip(mat, mat, -1);
		break;
	case 3://transpose+flip(0)=CCW
		transpose(mat, mat);
		flip(mat, mat, 0);
		break;
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////
// CChecker


//////////////////////////////////////////////////////////////////////////////////////////////
// CheckerDraw


void CCheckerDraw::
draw(cv::Mat & img)
{


	// color chart classic model
	CChartClassicModel cccm;
	cv::Mat lab; float J = 0; int N;
	std::vector<cv::Point2f>  fbox = cccm.box;
	std::vector<cv::Point2f> cellchart = cccm.cellchart;

	// tranformation
	cv::Matx33f ccT = cv::getPerspectiveTransform(fbox, m_pChecker->box);


	std::vector<cv::Point2f> bch(4), bcht(4);
	N = cellchart.size() / 4;
	for (size_t i = 0, k; i < N; i++)
	{

		k = 4 * i;
		bch[0] = cellchart[k + 0];
		bch[1] = cellchart[k + 1];
		bch[2] = cellchart[k + 2];
		bch[3] = cellchart[k + 3];

		for(int i=0;i<4;i++)
		{
			cout<<m_pChecker->box[i].x<<" "<<m_pChecker->box[i].y<<endl;
		}
		polyanticlockwise(bch);
		transform_points_forward(ccT, bch, bcht);

		cv::Point2f c(0, 0);
		for (size_t j = 0; j < 4; j++) 	c += bcht[j];  c /= 4;
		for (size_t j = 0; j < 4; j++) 	bcht[j] = ((bcht[j] - c)*0.50) + c;

		cv::line(img, bcht[0], bcht[1], m_color, m_thickness, LINE_AA);
		cv::line(img, bcht[1], bcht[2], m_color, m_thickness, LINE_AA);
		cv::line(img, bcht[2], bcht[3], m_color, m_thickness, LINE_AA);
		cv::line(img, bcht[3], bcht[0], m_color, m_thickness, LINE_AA);

	}

}


void CCheckerDraw::
transform_points_forward(const cv::Matx33f & T, const std::vector<cv::Point2f>& X, std::vector<cv::Point2f>& Xt)
{

	int N = X.size();
	Xt.clear(); Xt.resize(N);
	if (N == 0) return;

	cv::Matx31f p, xt;	cv::Point2f pt;
	for (size_t i = 0; i < N; i++)
	{
		p(0, 0) = X[i].x; p(1, 0) = X[i].y; p(2, 0) = 1;
		xt = T*p;
		pt.x = xt(0, 0) / xt(2, 0); pt.y = xt(1, 0) / xt(2, 0);
		Xt[i] = pt;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////
// CCheckerStreamIO

//(1) filename: nome do video.mp4
//(2) a_ratio : dimens�es do video
//(3) frame : o numero do frame na sequ�ncia do video
//(4) target : o tipo de chart que usamos(hoje estamos operando com MCC - 24 mas temos 2 outros formatos)
//(5) patch : o n�mero do patch dentro do chart
//(6) patch size : quantos pixels foram lidos dentro daquele patch(acho que sera um valor comum para todos, definido no inicio, mas precisamos desse registro em cada tupla)
//(7) color space : o espaco de cor no qual a informacao ser� provida
//(8) canal : a dimens�o dentro do espa�o de cor
//(9) average : a m�dia computada naquela dimens�o para o patch na tupla
//(10) stddev : o desvio padrao do valor medido dentro do referido patch, naquele canal
//(11) max : O m�ximo valor obtido para aquele patch, naquele espa�o de cor
//(12) min : O m�nimo valor obtido para aquele patch, naquele espa�o de cor


void
CCheckerStreamIO::open()
{

	on_stream.open(m_pathName.c_str(), std::fstream::out); // open file
	//on_stream.imbue(std::locale(std::cout.getloc(), new punct_facet<char, ','>)); // config
	//on_stream.imbue(std::locale(""));
	on_stream.precision(6);
	b_open_file = true;


}

void
CCheckerStreamIO::close()
{
	// close file
	on_stream.close();
	b_open_file = false;
}

void CCheckerStreamIO::
writeCSV(const CChecker& checker, const std::string & nameframe, cv::Size dim_frame_ratio,
	int itarget /*= 0*/, int iframe /*= 0*/)
{

	std::string strType[3] = { "MCC24", "SG140", "PASSPORT" };
	std::string strRgb[3] = { "R", "G", "B" };
	std::string strYCbCr[3] = { "Y", "Cb", "Cr" };
	std::string pathName;
	cv::Mat charts_rgb, charts_ycbcr;
	CChecker::TYPECHRT target;
	int N;

	// copy
	target = checker.target;
	charts_rgb   = checker.charts_rgb;
	charts_ycbcr = checker.charts_ycbcr;
	N = checker.N;


	// RGB color space
	for (size_t i = 0; i < N; i++)
	{
		for (size_t c = 0; c < 3; c++)
			on_stream
			<< nameframe << ';'								// filename
			<< dim_frame_ratio << ';' 						// a_ratio
			<< iframe << ';'								// frame
			<< strType[(int)target] << ';'					// target
			<< itarget << ';'								// target_no
			<< i + 1 << ';'									// patch
			<< charts_rgb.at<float>(3 * i + c, 0) << ';'	// p_size
			<< "RGB;"										// color space
			<< strRgb[c] << ';'								// canal
			<< charts_rgb.at<float>(3 * i + c, 1) << ';'	// average
			<< charts_rgb.at<float>(3 * i + c, 2) << ';'	// stddev
			<< charts_rgb.at<float>(3 * i + c, 4) << ';'	// max
			<< charts_rgb.at<float>(3 * i + c, 3)			// min
			<< endl;
	}


	// YCbCr color space
	for (size_t i = 0; i < N; i++)
	{
		for (size_t c = 0; c < 3; c++)
			on_stream
			<< nameframe << ';'								// filename
			<< dim_frame_ratio << ';' 						// a_ratio
			<< iframe << ';'								// frame
			<< strType[(int)target] << ';'					// target
			<< itarget << ';'								// target_no
			<< i + 1 << ';'									// patch
			<< charts_ycbcr.at<float>(3 * i + c, 0) << ';'	// p_size
			<< "YCbCr;"										// color space
			<< strYCbCr[c] << ';'							// canal
			<< charts_ycbcr.at<float>(3 * i + c, 1) << ';'	// average
			<< charts_ycbcr.at<float>(3 * i + c, 2) << ';'	// stddev
			<< charts_ycbcr.at<float>(3 * i + c, 4) << ';'	// max
			<< charts_ycbcr.at<float>(3 * i + c, 3)			// min
			<< endl;

	}


}

/**\brief write stram txt file
* format table
*       tabla
* |R|G|B|Y|Cb|Cr|
*
*/

void CCheckerStreamIO::
writeText(const CChecker & checker)
{
	cv::Mat charts_rgb, charts_ycbcr;
	CChecker::TYPECHRT target;
	int N;

	// copy
	target = checker.target;
	charts_rgb = checker.charts_rgb;
	charts_ycbcr = checker.charts_ycbcr;
	N = checker.N;

	std::string str;
	for (size_t i = 0; i < N; i++)
	{

		on_stream
			<< charts_rgb.at<float>(3 * i + 0, 1) << ';'
			<< charts_rgb.at<float>(3 * i + 1, 1) << ';'
			<< charts_rgb.at<float>(3 * i + 2, 1) << ';'
			<< charts_ycbcr.at<float>(3 * i + 0, 1) << ';'
			<< charts_ycbcr.at<float>(3 * i + 1, 1) << ';'
			<< charts_ycbcr.at<float>(3 * i + 2, 1)
			<< endl;
	}
}

}
}
