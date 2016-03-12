#include <precomp.hpp>



/* mkfilter -- given n, compute recurrence relation
   to implement Butterworth, Bessel or Chebyshev filter of order n
   A.J. Fisher, University of York   <fisher@minster.york.ac.uk>
   September 1992 */

/* Main module */

#include <stdio.h>
#include <cmath>
/* mkfilter -- given n, compute recurrence relation
   to implement Butterworth or Bessel filter of order n
   A.J. Fisher, University of York   <fisher@minster.york.ac.uk>
   September 1992 */

/* Header file */


/*
#define uint	    unsigned int
#define bool	    int
#define true	    1
#define false	    0
#define word	    int		    
*/
#define EPS	    1e-10

#define seq(s1,s2)  (strcmp(s1,s2) == 0)
#define unless(x)   if(!(x))
#define until(x)    while(!(x))
#define ifix(x)	    (int) (((x) >= 0.0) ? (x) + 0.5 : (x) - 0.5)


using namespace std;
using namespace cv::iirfilter;



IIRFilter::IIRFilter(string ftype, int o, double fs, vector<double> frequency)
{
    if (ftype != "butterworth")
    {
        cv::Exception e;
        e.code = -2;
        e.msg = "only butterworth filter is implemented";
    }
    // Frequency Bande passante où le gain du filtre est de 1 : 
    // 0 f1 passebas
    // f1 f2 passebande
    // f1 fs/2 passe haut
    if (frequency.size() != 2)
    {
        cv::Exception e;
        e.code = -1;
        e.msg = "you must give fLow and fHigh";

        throw e;

    }
    raw_alpha1 = frequency[0]/fs;
    raw_alpha2 = frequency[1]/fs;;
    if (abs(raw_alpha2-0.5)<0.000001)
        opts =  opt_hp|opt_bu; 
    else if (raw_alpha2==0)
        opts =  opt_lp|opt_bu; 
    else
        opts =  opt_bp|opt_bu; 

    order=o;
    compute_s();
    normalize();
    compute_z();
    expandpoly();
    int i;
    printf("Recurrence relation:\n");
    printf("y[n] = ");
    b.resize(numpoles+1);
    a.resize(numpoles+1);
    for (i=0; i < numpoles+1; i++)
    { 
        if (i > 0) printf("     + ");
        printf("(%3g * x[n-%2d])\n", xcoeffs[i], numpoles-i);
        b[ numpoles-i]= xcoeffs[i]/abs(fc_gain);
    }
    putchar('\n');
    for (i=0; i < numpoles; i++)
    { 
        printf("     + (%14.10f * y[n-%2d])\n", ycoeffs[i], numpoles-i);
        a[numpoles-i]=ycoeffs[i];
    }
    a[0]=0;
    putchar('\n');


}




complex<double>   IIRFilter::eval(complex<double>   coeffs[], int np, complex<double>   z)
  { /* evaluate polynomial in z, substituting for z */
    complex<double>   sum(0,0); int i;
    for (i=np; i >= 0; i--) sum = sum* z+ coeffs[i];
    return sum;
  }

complex<double>  IIRFilter::evaluate(complex<double>  topco[],complex<double>  botco[], int np, complex<double>  z)
  { /* evaluate response, substituting for z */
    return eval(topco, np, z)/ eval(botco, np, z);
  }

complex<double>  bessel_poles[] =
  { /* table produced by /usr/fisher/bessel --	N.B. only one member of each C.Conj. pair is listed */
    complex<double> ( -1.000000e+00,  0.000000e+00 ),	 complex<double> ( -8.660254e-01, -5.000000e-01 ),	 complex<double> ( -9.416000e-01,	0.000000e+00),	 
    complex<double> ( -7.456404e-01, -7.113666e-01 ),	 complex<double> (-9.047588e-01, -2.709187e-01 ),	     complex<double> ( -6.572112e-01, -8.301614e-01 ),
    complex<double> (-9.264421e-01,  0.000000e+00 ),	 complex<double> ( -8.515536e-01, -4.427175e-01 ),	 complex<double> ( -5.905759e-01, -9.072068e-01 ),	 
    complex<double> (-9.093907e-01, -1.856964e-01 ),	 complex<double> ( -7.996542e-01, -5.621717e-01 ),	 complex<double> ( -5.385527e-01, -9.616877e-01),
    complex<double> ( -9.194872e-01,  0.000000e+00 ),	 complex<double> ( -8.800029e-01, -3.216653e-01 ),	 complex<double> ( -7.527355e-01, -6.504696e-01 ),
    complex<double> ( -4.966917e-01, -1.002509e+00 ),	 complex<double> ( -9.096832e-01, -1.412438e-01 ),	 complex<double> (-8.473251e-01, -4.259018e-01 ),	 
    complex<double> ( -7.111382e-01, -7.186517e-01 ),	 complex<double> ( -4.621740e-01, -1.034389e+00 ),	 complex<double> ( -9.154958e-01,	0.000000e+00 ),	 
    complex<double> ( -8.911217e-01, -2.526581e-01 ),	 complex<double> ( -8.148021e-01, -5.085816e-01),	 complex<double> ( -6.743623e-01, -7.730546e-01 ),	 
    complex<double> ( -4.331416e-01, -1.060074e+00 ),	 complex<double> ( -9.091347e-01, -1.139583e-01 ),	 complex<double> ( -8.688460e-01, -3.430008e-01 ),
    complex<double> ( -7.837694e-01, -5.759148e-01 ),	 complex<double> ( -6.417514e-01, -8.175836e-01 ),	 complex<double> ( -4.083221e-01, -1.081275e+00 )
  };

complex<double>  cmone(-1.0, 0.0);
complex<double>  czero(0.0, 0.0);
complex<double>  cone(1.0, 0.0);
complex<double>  ctwo(2.0, 0.0);
complex<double>  chalf(0.5, 0.0);



int decodeopts();
double getfarg();

#define cneg(z) csub(czero, z)



void IIRFilter::choosepole(complex<double>  z)
  { if (z.real() < 0.0)
      { if (polemask & 1) spoles[numpoles++] = z;
	//polemask >>= 1;
      }
  }





void  IIRFilter::compute_s() /* compute S-plane poles for prototype LP filter */
  { 
    polemask=1;
    numpoles =0;
    if (opts & opt_be)
      { /* Bessel filter */
	int i;
	int p = (order*order)/4; /* ptr into table */
	if (order & 1) choosepole(bessel_poles[p++]);
	for (i=0; i < order/2; i++)
	  { choosepole(bessel_poles[p]);
	    choosepole(conj(bessel_poles[p]));
	    p++;
	  }
      }
    if (opts & (opt_bu | opt_ch))
      { /* Butterworth filter */
	int i;
	for (i=0; i < 2*order; i++)
	  { complex<double>  s;
	    s = complex<double> (0,(order & 1) ? (i*PI) / order : ((i+0.5)*PI) / order);
	    choosepole(exp(s));
	  }
      }
    if (opts & opt_ch)
      { /* modify for Chebyshev (p. 136 DeFatta et al.) */
	double rip, eps, y; int i;
	if (chebrip >= 0.0)
	  { fprintf(stderr, "mkfilter: Chebyshev ripple is %g dB; must be .lt. 0.0\n", chebrip);
	    exit(1);
	  }
	rip = pow(10.0, -chebrip / 10.0);
	eps = sqrt(rip - 1.0);
	y = asinh(1.0 / eps) / (double) order;
	if (y <= 0.0)
	  { fprintf(stderr, "mkfilter: bug: Chebyshev y=%g; must be .gt. 0.0\n", y);
	    exit(1);
	  }
	for (i=0; i < numpoles; i++)
	  { 
          spoles[i] = complex<double> (spoles[i].real() * sinh(y),spoles[i].imag()* cosh(y));
	  }
      }
  }


void IIRFilter::normalize()
  { complex<double>  w1, w2; int i;
    /* for bilinear transform, perform pre-warp on alpha values */
    if (opts & opt_w)
      { warped_alpha1 = raw_alpha1;
	warped_alpha2 = raw_alpha2;
      }
    else
      { warped_alpha1 = tan(PI * raw_alpha1) / PI;
	warped_alpha2 = tan(PI * raw_alpha2) / PI;
      }
    w1=complex<double> (TWOPI * warped_alpha1, 0.0);
    w2=complex<double> (TWOPI * warped_alpha2, 0.0);
    /* transform prototype into appropriate filter type (lp/hp/bp) */
    switch (opts & (opt_lp + opt_hp + opt_bp))
      { case opt_lp:
	    for (i=0; i < numpoles; i++) spoles[i] = spoles[i]* w1;
	    break;

	case opt_hp:
	    for (i=0; i < numpoles; i++) spoles[i] = w1/ spoles[i];
	    /* also N zeros at (0,0) */
	    break;

	case opt_bp:
	  { complex<double>  w0, bw;
	    w0 = sqrt(w1* w2);
	    bw = w2- w1;
	    for (i=0; i < numpoles; i++)
	      { complex<double>  hba, temp;
		hba = chalf*spoles[i]* bw;
		temp = w0/ hba;
		temp = sqrt(cone-temp* temp);
		spoles[i] = (hba*( cone+ temp));
		spoles[numpoles+i] = hba* (cone- temp);
	      }
	    /* also N zeros at (0,0) */
	    numpoles *= 2;
	    break;
	  }
      }
  }

void IIRFilter::compute_z() /* given S-plane poles, compute Z-plane poles */
  { int i;
    for (i=0; i < numpoles; i++)
      { /* use bilinear transform */
	complex<double>  top, bot;
	top = ctwo+ spoles[i];
	bot = ctwo- spoles[i];
	zpoles[i] = top/bot;
	switch (opts & (opt_lp + opt_hp + opt_bp))
	  { case opt_lp:    zzeros[i] = cmone; break;
	    case opt_hp:    zzeros[i] = cone; break;
	    case opt_bp:    zzeros[i] = (i & 1) ? cone : cmone; break;
	  }
      }
  }


void IIRFilter::multin(complex<double>  w, complex<double>  coeffs[])
  { /* multiply factor (z-w) into coeffs */
    complex<double>  nw; 
    int i;
    nw = -w;
    for (i=numpoles; i >= 1; i--)
      coeffs[i] = (nw* coeffs[i])+coeffs[i-1];
    coeffs[0] = nw* coeffs[0];
  }

void IIRFilter::expand(complex<double>  pz[],complex<double>   coeffs[])
  { /* compute product of poles or zeros as a polynomial of z */
    int i;
    coeffs[0] = cone;
    for (i=0; i < numpoles; i++) coeffs[i+1] = czero;
    for (i=0; i < numpoles; i++) multin(pz[i], coeffs);
    /* check computed coeffs of z^k are all real */
    for (i=0; i < numpoles+1; i++)
      { if (abs(coeffs[i].imag()) > EPS)
	  { fprintf(stderr, "mkfilter: coeff of z^%d is not real; poles are not complex conjugates\n", i);
	    exit(1);
	  }
      }
  }

void IIRFilter::expandpoly() /* given Z-plane poles & zeros, compute top & bot polynomials in Z, and then recurrence relation */
  { complex<double>  topcoeffs[MAXPOLES+1], botcoeffs[MAXPOLES+1];
    complex<double>  st, zfc; int i;
    expand(zzeros, topcoeffs);
    expand(zpoles, botcoeffs);
    dc_gain = evaluate(topcoeffs, botcoeffs, numpoles, cone);
    st=    complex<double> (0.0, TWOPI * 0.5 * (raw_alpha1 + raw_alpha2)); /* "jwT" for centre freq. */

    zfc = exp(st);
    fc_gain = evaluate(topcoeffs, botcoeffs, numpoles, zfc);
    hf_gain = evaluate(topcoeffs, botcoeffs, numpoles, cmone);
    for (i=0; i <= numpoles; i++)
      { xcoeffs[i] = topcoeffs[i].real() / botcoeffs[numpoles].real();
	ycoeffs[i] = -(botcoeffs[i].real() / botcoeffs[numpoles].real());
      }
  }




void IIRFilter::printrecurrence() /* given (real) Z-plane poles & zeros, compute & print recurrence relation */
  { int i;
    printf("Recurrence relation:\n");
    printf("y[n] = ");
    for (i=0; i < numpoles+1; i++)
      { if (i > 0) printf("     + ");
	printf("(%3g * x[n-%2d])\n", xcoeffs[i], numpoles-i);
      }
    putchar('\n');
    for (i=0; i < numpoles; i++)
      { printf("     + (%14.10f * y[n-%2d])\n", ycoeffs[i], numpoles-i);
      }
    putchar('\n');
  }


void IIRFilter::printfilter()
{ 
  }
