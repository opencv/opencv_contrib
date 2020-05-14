#include "opencv2/mcc/graph_cluster.hpp"
#include "opencv2/mcc/core.hpp"

namespace cv{
namespace mcc{
CB0cluster::CB0cluster()
{

}


CB0cluster::~CB0cluster()
{

}


void CB0cluster::
group()
{

	int n = X.size();
	G.clear(); G.resize(n);

	for (size_t i = 0; i < n - 1; i++)
	{
		std::vector<float> Y;
		Y.clear(); Y.resize(n - i);
		Y[0] = 0;

		// 1. semejanza intergrupo
		float dist, w, y;
		for (size_t j = i + 1, k = 1; j < n; j++, k++)
		{
			//dist(X_i,X_j)
			dist = norm(X[i] - X[j]);

			//\frac{|W_i - W_j|}{W_i + W_j}
			w = abs(W[i] - W[j]) / (W[i] + W[j]);
			w = (w < 0.3);

			y = w*dist;
			Y[k] = (y<B0[i])*y;

		}


		// 2. Buscar los b0-semejantes
		if (!G[i]) G[i] = i + 1;

		std::vector<int> pos_b0;
		find(Y, pos_b0);

		// 3. Analisis de la casuistica

		int m = pos_b0.size();
		if (!m) continue;

		//reajuste de las coordenadas
		std::vector<int> pos_nz, pos_z;
		for (size_t j = 0; j < m; j++)
		{
			pos_b0[j] = pos_b0[j] + i;
			if (G[pos_b0[j]]) pos_nz.push_back(j);
			else pos_z.push_back(j);
		}

		//marcados
		for (size_t j = 0; j < pos_z.size(); j++)
		{
			pos_z[j] = pos_b0[pos_z[j]];
			G[pos_z[j]] = G[i];

		}


		//no marcados
		//Se unir�n todos los grupos presentes en
		//G(j) con los de G(i) y el representante
		//ser� el menor de todos
		//UNIR(G(i), G(j));

		if (!pos_nz.size()) continue;


		std::vector<int> g;
		for (size_t j = 0; j < pos_nz.size(); j++)
		{
			pos_nz[j] = pos_b0[pos_nz[j]];
			g.push_back(G[pos_nz[j]]);
		}

		unique(g, g);
		for (size_t k = 0; k < g.size(); k++)
		{
			int gk = g[k];
			for (size_t j = 0; j < G.size(); j++)
				if (G[j] == gk)G[j] = G[i];
		}
	}

	//ultimo caso
	if (!G[n - 1]) G[n - 1] = n;

	//formar grupos
	//Organizamos los  grupos de manera tal que exista una correspodencia con
	//el eje num�rico

	std::vector<int> S;
	S = G; unique(S, S);
	for (size_t k = 0; k < S.size(); k++)
	{
		int gk = S[k];
		for (size_t j = 0; j < G.size(); j++)
			if (G[j] == gk)G[j] = k;
	}


}

}
}
