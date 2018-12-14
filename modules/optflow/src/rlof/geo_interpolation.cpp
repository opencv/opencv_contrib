// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// This functions have been contributed by Jonas Geisters <geistert@nue.tu-berlin.de>

#include "../precomp.hpp"

#include "geo_interpolation.hpp"
#include <string>
#include <map>
namespace cv {
namespace optflow {

struct Graph_helper {
    std::vector<int> mem;
    int e_size;
    Graph_helper(int k, int num_nodes) {
        e_size = (2 * k + 1);
        mem.resize(e_size * num_nodes, 0);
    }
    inline int size(int id) {
        int r_addr = id * (e_size);
        return mem[r_addr];
    }
    inline int * data(int id) {
        int r_addr = id * (e_size)+1;
        return &mem[r_addr];
    }
    inline void add(int id, std::pair<float, int> data) {
        int r_addr = id * (e_size);
        int size = ++mem[r_addr];
        r_addr += 2 * size - 1;//== 1 + 2*(size-1);
        *(float*)&mem[r_addr] = data.first;
        mem[r_addr + 1] = data.second;
    }
    inline bool color_in_target(int id, int color) {
        int r_addr = id * (e_size);
        int size = mem[r_addr];
        r_addr += 2;
        for (int i = 0; i < size; i++) {
            if (mem[r_addr] == color) {
                return true;
            }
            r_addr += 2;
        }
        return false;
    }

};

Mat sgeo_dist(const Mat& gra, int y, int x, float max, Mat &prev)
{
    std::vector <Point2f> points;
    points.push_back(Point2f(static_cast<float>(x), static_cast<float>(y)));
    return sgeo_dist(gra, points, max, prev);
}
Mat sgeo_dist(const Mat& gra, const std::vector<Point2f> & points, float max, Mat &prev)
{
    int Dx[] = { -1,0,1,-1,1,-1,0,1 };
    int Dy[] = { -1,-1,-1,0,0,1,1,1 };
    Mat dm(gra.rows, gra.cols, CV_32F, Scalar(max));
    prev = Mat(gra.rows, gra.cols, CV_8U, Scalar(255));

    std::multimap<float, Vec2i > not_visited_with_value;

    for (auto i = points.begin(); i != points.end(); i++)
    {
        int y = static_cast<int>(i->y);
        int x = static_cast<int>(i->x);
        not_visited_with_value.insert(std::pair<float, Vec2i >(0.f, Vec2i(y, x)));
        dm.at<float>(y, x) = 0;
    }

    bool done = false;
    while (!done)
    {
        if (not_visited_with_value.begin() == not_visited_with_value.end()) {
            done = true;
            break;
        }
        std::multimap<float, Vec2i >::iterator current_it = not_visited_with_value.begin();
        std::pair<float, Vec2i > current_p = *current_it;
        not_visited_with_value.erase(current_it);

        int y = current_p.second[0];
        int x = current_p.second[1];
        float cur_d = current_p.first;

        if (dm.at<float>(y, x) != cur_d) {
            continue;
        }

        Vec8f gra_e = gra.at<Vec8f>(y, x);

        for (int i = 0; i < 8; i++) {
            if (gra_e[i] < 0) {
                continue;
            }
            int dx = Dx[i];
            int dy = Dy[i];

            if (dm.at<float>(y + dy, x + dx) > cur_d + gra_e[i]) {
                dm.at<float>(y + dy, x + dx) = cur_d + gra_e[i];
                prev.at<uchar>(y + dy, x + dx) = static_cast<uchar>(7 - i);
                not_visited_with_value.insert(std::pair<float, Vec2i >(cur_d + gra_e[i], Vec2i(y + dy, x + dx)));
            }

        }
    }




    return dm;
}

Mat interpolate_irregular_nn_raster(const std::vector<Point2f> & prevPoints,
    const std::vector<Point2f> & nextPoints,
    const std::vector<uchar> & status,
    const Mat & i1)
{
    Mat gra = getGraph(i1, 0.1f);
    int Dx[] = { -1,0,1,-1,1,-1,0,1 };
    int Dy[] = { -1,-1,-1,0,0,1,1,1 };
    int max_rounds = 10;
    Mat dirt = Mat(gra.rows, gra.cols, CV_8U, Scalar(0));
    Mat quellknoten = Mat(gra.rows, gra.cols, CV_32S, Scalar(-1));
    Mat dist = Mat(gra.rows, gra.cols, CV_32F, Scalar(std::numeric_limits<float>::max()));
    /*
        * assign quellknoten ids.
        */
    for (int i = 0; i < static_cast<int>(prevPoints.size()); i++)
    {
        int x = (int)prevPoints[i].x;
        int y = (int)prevPoints[i].y;
        if (status[i] == 0)
            continue;
        dirt.at<uchar>(y, x) = 1;
        dist.at<float>(y, x) = 0;
        quellknoten.at<int>(y, x) = i;
    }

    bool clean = true;
    bool done = false;
    int x = 0;
    int y = 0;
    int rounds = 0;
    while (!done) {
        /*
            * Update x and y
            * on even rounds go rasterscanorder , on odd round inverse rasterscanorder
            */
        if (rounds % 2 == 0) {
            x++;
            if (x >= gra.cols) {
                x = 0;
                y++;
                if (y >= gra.rows) {
                    y = 0;
                    rounds++;
                    y = gra.rows - 1;
                    x = gra.cols - 1;
                    if (rounds >= max_rounds || clean) {
                        done = true;
                        break;
                    }
                }
            }
        }
        else {
            x--;
            if (x < 0) {
                x = gra.cols - 1;
                y--;
                if (y < 0) {
                    y = gra.rows - 1;
                    rounds++;
                    y = 0;
                    x = 0;
                    if (rounds >= max_rounds || clean) {
                        done = true;
                        break;
                    }
                }
            }
        }
        if (dirt.at<uchar>(y, x) == 0) {
            continue;
        }
        dirt.at<uchar>(y, x) = 0;

        float c_dist = dist.at<float>(y, x);
        Vec8f gra_e = gra.at<Vec8f>(y, x);

        for (int i = 0; i < 8; i++) {
            int tx = Dx[i];
            int ty = Dy[i];
            if (ty == 0 && tx == 0) {
                continue;
            }
            if (x + tx < 0 || x + tx >= gra.cols) {
                continue;
            }
            if (y + ty < 0 || y + ty >= gra.rows) {
                continue;
            }
            if (c_dist > dist.at<float>(y + ty, x + tx)) {
                if (c_dist > dist.at<float>(y + ty, x + tx) + gra_e[i]) {
                    quellknoten.at<int>(y, x) = quellknoten.at<int>(y + ty, x + tx);
                    dist.at<float>(y, x) = dist.at<float>(y + ty, x + tx) + gra_e[i];
                    dirt.at<uchar>(y, x) = 1;
                    clean = false;
                }
            }
            else {
                if (c_dist + gra_e[i] < dist.at<float>(y + ty, x + tx)) {
                    quellknoten.at<int>(y + ty, x + tx) = quellknoten.at<int>(y, x);
                    dist.at<float>(y + ty, x + tx) = dist.at<float>(y, x) + gra_e[i];
                    dirt.at<uchar>(y + ty, x + tx) = 1;
                    clean = false;
                }
            }
        }



    }

    Mat nnFlow(i1.rows, i1.cols, CV_32FC2, Scalar(0));
    for (y = 0; y < i1.rows; y++) {
        for (x = 0; x < i1.cols; x++) {

            int id = quellknoten.at<int>(y, x);
            if (id != -1)
            {
                nnFlow.at<Point2f>(y, x) = nextPoints[id] - prevPoints[id];
            }
        }
    }
    return nnFlow;
}

Mat interpolate_irregular_knn(
    const std::vector<Point2f> & _prevPoints,
    const std::vector<Point2f> & _nextPoints,
    const std::vector<uchar> & status,
    const Mat &color_img,
    int k,
    float pixeldistance)
{
    Mat in(color_img.rows, color_img.cols, CV_32FC2);
    Mat mask = Mat::zeros(color_img.rows, color_img.cols, CV_8UC1);

    for (unsigned n = 0; n < _prevPoints.size(); n++)
    {
        if (_prevPoints[n].x >= 0 && _prevPoints[n].y >= 0 && _prevPoints[n].x < color_img.cols && _prevPoints[n].y < color_img.rows)
        {
            in.at<Point2f>(_prevPoints[n]) = _nextPoints[n] - _prevPoints[n];
            mask.at<uchar>(_prevPoints[n]) = status[n];
        }

    }
    int Dx[] = { -1,0,1,-1,1,-1,0,1 };
    int Dy[] = { -1,-1,-1,0,0,1,1,1 };
    Mat gra = getGraph(color_img, pixeldistance);
    Mat nnFlow(in.rows, in.cols, CV_32FC2, Scalar(0));

    std::multimap<float, Vec2i > my_agents; // <arrivaltim , < target, color >>
    Graph_helper graph_helper(k, in.rows*in.cols); //< arrivaltime, color>


    int color = 0;
    std::vector<Vec2i> flow_point_list;
    for (int y = 0; y < in.rows; y++) {
        for (int x = 0; x < in.cols; x++) {
            if (mask.at<uchar>(y, x) > 0) {
                flow_point_list.push_back(Vec2i(y, x));
                nnFlow.at<Vec2f>(y, x) = in.at<Vec2f>(y, x);

                int v_id = (y * in.cols + x);
                graph_helper.add(v_id, std::pair<float, int>(0.f, color));


                Vec8f gra_e = gra.at<Vec8f>(y, x);
                for (int i = 0; i < 8; i++) {
                    if (gra_e[i] < 0)
                        continue;
                    int dx = Dx[i];
                    int dy = Dy[i];
                    int target = (y + dy) * in.cols + (x + dx);
                    Vec2i agent(target, color);
                    my_agents.insert(std::pair<float, Vec2i >(gra_e[i], agent));

                }
                color++;
            }
        }
    }

    int global_time = 0;

    bool done = false;
    while (!done) {
        if (my_agents.size() == 0) {
            done = true;
            break;
        }
        global_time++;

        std::multimap<float, Vec2i >::iterator current_it = my_agents.begin();
        std::pair<float, Vec2i > current_p = *current_it;
        my_agents.erase(current_it);

        int target = current_p.second[0];
        color = current_p.second[1];
        float arriv_time = current_p.first;

        Vec8f gra_e = gra.at<Vec8f>(target);// (y*cols+x)
        if (graph_helper.size(target) >= k) {
            continue;
        }

        bool color_found_in_target = graph_helper.color_in_target(target, color);
        if (color_found_in_target) {
            continue;
        }
        graph_helper.add(target, std::pair<float, int>(arriv_time, color));


        for (int i = 0; i < 8; i++) {
            if (gra_e[i] < 0)
                continue;

            int dx = Dx[i];
            int dy = Dy[i];
            int new_target = target + dx + (dy*in.cols);
            if (graph_helper.size(new_target) >= k) {
                continue;
            }
            color_found_in_target = graph_helper.color_in_target(new_target, color);
            if (color_found_in_target) {
                continue;
            }
            Vec2i new_agent(new_target, color);
            my_agents.insert(std::pair<float, Vec2i >(arriv_time + gra_e[i], new_agent));

        }
    }

    Mat ret(in.rows, in.cols*k, CV_32FC2);
    for (int y = 0; y < in.rows; y++) {
        for (int x = 0; x < in.cols; x++) {
            for (int i = 0; i < k; i++) {
                float dist = *((float*)(graph_helper.data(y*in.cols + x) + 2 * i));
                float id = *((float*)(graph_helper.data(y*in.cols + x) + 2 * i + 1));
                ret.at<Vec2f>(y, k*x + i) = Vec2f(dist, id);
            }
        }
    }
    return ret;
}

Mat getGraph(const Mat &image, float edge_length)
{

    int Dx[] = { -1,0,1,-1,1,-1,0,1 };
    int Dy[] = { -1,-1,-1,0,0,1,1,1 };
    Mat gra(image.rows, image.cols, CV_32FC(8));

    for (int y = 0; y < gra.rows; y++) {
        for (int x = 0; x < gra.cols; x++) {

            for (int i = 0; i < 8; i++) {
                int dx = Dx[i];
                int dy = Dy[i];
                gra.at<Vec8f>(y, x)[i] = -1;

                if (x + dx < 0 || y + dy < 0 || x + dx >= gra.cols || y + dy >= gra.rows) {
                    continue;
                }

                if (i < 4) {
                    int si = 7 - i;
                    gra.at<Vec8f>(y, x)[i] = gra.at<Vec8f>(y + dy, x + dx)[si];
                }
                else {
                    float p1 = dx * dx*edge_length*edge_length + dy * dy*edge_length*edge_length;
                    float p2 = static_cast<float>(image.at<Vec3b>(y, x)[0] - image.at<Vec3b>(y + dy, x + dx)[0]);
                    float p3 = static_cast<float>(image.at<Vec3b>(y, x)[1] - image.at<Vec3b>(y + dy, x + dx)[1]);
                    float p4 = static_cast<float>(image.at<Vec3b>(y, x)[2] - image.at<Vec3b>(y + dy, x + dx)[2]);
                    gra.at<Vec8f>(y, x)[i] = sqrt(p1 + p2 * p2 + p3 * p3 + p4 * p4);
                }

            }

        }
    }

    return gra;
}


Mat interpolate_irregular_nn(
    const std::vector<Point2f> & _prevPoints,
    const std::vector<Point2f> & _nextPoints,
    const std::vector<uchar> & status,
    const Mat &color_img,
    float pixeldistance)
{
    int Dx[] = { -1,0,1,-1,1,-1,0,1 };
    int Dy[] = { -1,-1,-1,0,0,1,1,1 };
    std::vector<Point2f> prevPoints, nextPoints;
    std::map<std::pair<float, float>, std::pair<float, float>> flowMap;
    for (unsigned n = 0; n < _prevPoints.size(); n++)
    {
        if (status[n] != 0)
        {
            flowMap.insert(std::make_pair(
                std::make_pair(_prevPoints[n].x, _prevPoints[n].y),
                std::make_pair(_nextPoints[n].x, _nextPoints[n].y)));
            prevPoints.push_back(_prevPoints[n]);
            nextPoints.push_back(_nextPoints[n]);
        }

    }

    Mat gra = getGraph(color_img, pixeldistance);

    Mat prev;
    Mat geo_dist = sgeo_dist(gra, prevPoints, std::numeric_limits<float>::max(), prev);


    Mat nnFlow = Mat::zeros(color_img.size(), CV_32FC2);
    for (int y = 0; y < nnFlow.rows; y++)
    {
        for (int x = 0; x < nnFlow.cols; x++)
        {
            int cx = x;
            int cy = y;
            while (prev.at<uchar>(cy, cx) != 255)
            {
                int i = prev.at<uchar>(cy, cx);
                cx += Dx[i];
                cy += Dy[i];
            }
            auto val = flowMap[std::make_pair(static_cast<float>(cx), static_cast<float>(cy))];
            nnFlow.at<Vec2f>(y, x) = Vec2f(val.first - cx, val.second - cy);
        }
    }
    return nnFlow;
}

}} // namespace
