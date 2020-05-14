#ifndef _MCC_CHECKER_MODEL_H
#define _MCC_CHECKER_MODEL_H

#include "core.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>


namespace cv{
namespace mcc{


	/** CChartClassicModel
	  *
	  * @brief ColorChecker� Classic
	  * @note
	  * O alvo ColorChecker� Classic � um quadro com um conjunto de 24 amostras
	  * de cores naturais, espectrais, prim�rias e cinzas, cientificamente
	  * produzidas em uma ampla gama de tonalidades. Muitos desses quadrados
	  * representam as cores de objetos naturais, tais como as cores da pele
	  * humana, da vegeta��o e do c�u azul. Visto que exemplificam as cores dos
	  * seus correspondentes no mundo real e refletem a luz da mesma forma em
	  * todas as regi�es do espectro vis�vel, os quadrados matizar�o as cores dos
	  * objetos naturais que representam sob qualquer ilumina��o e com qualquer
	  * processo de reprodu��o de cores.O ColorChecker Classic pode tamb�m ser
	  * utilizado para criar um balan�o de branco com sua c�mera digital para
	  * garantir um branco neutro, preciso e uniforme sob qualquer condi��o
	  * de ilumina��o.
	  *
	  * O ColorChecker est� dispon�vel em dois tamanhos : Classic(8, 25 x 11 pol.)
	  * e Mini(2, 25 x 3, 25 pol.).Guarde um no seu est�dio e leve o outro no seu
	  * bolso para fotografias na loca��o. Caso deseje criar perfis de cores para
	  * sua c�mera digital e dar um passo a frente em termos de qualidade das suas
	  * fotos digitais, experimente usar o ColorChecker Classic.
	  *
	  * @autor Pedro Marrero Fern�ndez
	  */

	class CChartClassicModel
	{

	public:

		typedef struct
		_SUBCCMModel {

			cv::Mat sub_chart;
			cv::Size2i color_size;
			std::vector<cv::Point2f> centers;

		}SUBCCMModel;

	public:

		CChartClassicModel();
		~CChartClassicModel();

		/** @brief evaluate submodel in this checker type*/
		bool evaluate(const SUBCCMModel &subModel, int &offset, int &iTheta, float &error);

		// function utils

		void rotate90();
		void copyToColorMat(cv::Mat &lab, int cs = 0);


	public:

		// Cie L*a*b* values use illuminant D50 2 degree observer sRGB values for
		// for iluminante D65.
		static const float chart[24][9];

		cv::Size2i size;
		cv::Size2f boxsize;
		std::vector<cv::Point2f> box;
		std::vector<cv::Point2f> cellchart;
		std::vector<cv::Point2f> center;


	private:

		/** \brief match checker color
		  * \param[in] subModel sub-checker
		  * \param[in] iTheta angle
		  * \param[out] error
		  * \param[out] ierror
		  * \return state
		  */
		bool match(const SUBCCMModel &subModel, int iTheta, float &error, int &ierror);

		/** \brief euclidian dist L2 for Lab space
		  * \note
		  * \f$ \sum_i \sqrt (\sum_k (ab1-ab2)_k.^2) \f$
		  * \param[in] lab1
		  * \param[in] lab2
		  * \return distance
		  */
		float dist_color_lab(const cv::Mat &lab1, const cv::Mat &lab2);

		/** \brief rotate matrix 90 grado */
		void rot90(cv::Mat &mat, int itheta);

	};




	/** CChecker
	  *
	  * \brief checker model
	  * \autor Pedro Marrero Fernandez
	  *
	  */
	class CChecker
	{

	public:

		typedef
		enum _TYPECHAR
		{
			MCC24 = 0,
			SG140,
			PASSPORT

		}TYPECHRT;

	public:

		CChecker(): target(MCC24), N(24) {}
		~CChecker() {}



	public:

		TYPECHRT target;				// o tipo de checkercolor
		int N;							// number of charts
		std::vector< cv::Point2f > box;	// corner box
		cv::Mat charts_rgb;				// charts profile rgb color space
		cv::Mat charts_ycbcr;			// charts profile YCbCr color space
		float cost;						// cost to aproximate
		cv::Point2f center;


	};


	/** \brief checker draw
	  * \autor Pedro Marrero Fernandez
	  */
	class CCheckerDraw
	{

	public:

		CCheckerDraw(CChecker *pChecker, cv::Scalar color = CV_RGB(0, 250, 0), int thickness = 2)
			:m_pChecker(pChecker)
			,m_color(color)
			,m_thickness(thickness)
		{
			assert(pChecker);
		}

		void draw(cv::Mat &img);

	private:
		CChecker *m_pChecker;
		cv::Scalar m_color;
		int m_thickness;

	private:
		/** \brief transformation perspetive*/
		void transform_points_forward(
			const cv::Matx33f &T,
			const std::vector<cv::Point2f> &X,
			std::vector<cv::Point2f> &Xt
			);

	};


	/** \brief checker stream
	  * \autor Pedro Marrero Fernandez
	  */
	class CCheckerStreamIO
	{

		template <class charT, charT sep>
		class punct_facet : public std::numpunct<charT> {
		protected:
			charT do_decimal_point() const { return sep; }
		};


	public:

		CCheckerStreamIO(std::string pathName )
		: m_pathName(pathName)
		, b_open_file(false)
		{

		}

		~CCheckerStreamIO()
		{
			if(b_open_file)
			close();
		}

		void open();
		void close();

		void createHeaderCsv() {

			on_stream << "filename;a_ratio;frame;target;target_no;patch;p_size;space;channel;average;stddev;max;min" << endl; // header table
		}

		/**\brief write stream csv file
		  * format table charts information:
		  *						Tabla
		  *		  |filename|a_ratio|frame|target|target_no|patch|p_size|average|stddev|max|min|
		  *	RGB   |		   |	   |			|		  |     |      |       |      |   |   |
		  * YCbCr |		   |	   |			|         |     |      |       |      |   |   |
		  *
		  * (1) filename: nome do video.mp4
		  * (2) a_ratio : dimens�es do video
		  * (3) frame : o numero do frame na sequ�ncia do video
		  * (4) target : o tipo de chart que usamos(hoje estamos operando com MCC - 24 mas temos 2 outros formatos)
		  * (5) target_no
		  * (6) patch : o n�mero do patch dentro do chart
		  * (7) patch size : quantos pixels foram lidos dentro daquele patch(acho que sera um valor comum para todos, definido no inicio, mas precisamos desse registro em cada tupla)
		  * (8) color space : o espaco de cor no qual a informacao ser� provida
		  * (9) canal : a dimens�o dentro do espa�o de cor
		  * (10) average : a m�dia computada naquela dimens�o para o patch na tupla
		  * (11) stddev : o desvio padrao do valor medido dentro do referido patch, naquele canal
		  * (12) max : O m�ximo valor obtido para aquele patch, naquele espa�o de cor
		  * (13) min : O m�nimo valor obtido para aquele patch, naquele espa�o de cor
		  *
		  */
		void writeCSV(const CChecker& checker, const std::string & nameframe, cv::Size dim_frame_ratio, int itarget = 0, int iframe = 0);

		/**\brief write stram txt file
		  * format table
		  *       tabla
		  * |R|G|B|Y|Cb|Cr|
		  *
		  */
		void writeText(const CChecker& checker);

	public:

		std::string m_pathName;
		std::ofstream on_stream;
		bool b_open_file;




	};


}
}

#endif //_MCC_CHECKER_MODEL_H
