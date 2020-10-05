//
// Created by YIMIN TANG on 9/18/20.
//

#ifndef OPENCV_CONCAVEMAN_HPP
#define OPENCV_CONCAVEMAN_HPP
//
// Author: Stanislaw Adaszewski, 2019
// C++ port from https://github.com/mapbox/concaveman (js)
//
// Comments from js repo added by wheeled
//

#pragma once

#include <memory>
#include <stdexcept>
#include <list>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <limits>
#include <set>
#include <queue>
#include <assert.h>

//#define DEBUG // uncomment to dump debug info to screen
//#define DEBUG_2 // uncomment to dump second-level debug info to screen

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


template<class T> class compare_first {
public:
    bool operator()(const T &a, const T &b) {
        return (std::get<0>(a) < std::get<0>(b));
    }
};


template<class T> T orient2d(
        const std::array<T, 2> &p1,
        const std::array<T, 2> &p2,
        const std::array<T, 2> &p3) {

    T res = (p2[1] - p1[1]) * (p3[0] - p2[0]) -
            (p2[0] - p1[0]) * (p3[1] - p2[1]);

    return res;
}


// check if the edges (p1,q1) and (p2,q2) intersect
template<class T> bool intersects(
        const std::array<T, 2> &p1,
        const std::array<T, 2> &q1,
        const std::array<T, 2> &p2,
        const std::array<T, 2> &q2) {

    auto res = (p1[0] != q2[0] || p1[1] != q2[1]) &&
               (q1[0] != p2[0] || q1[1] != p2[1]) &&
               (orient2d(p1, q1, p2) > 0) != (orient2d(p1, q1, q2) > 0) &&
               (orient2d(p2, q2, p1) > 0) != (orient2d(p2, q2, q1) > 0);

    return res;
}


// square distance between 2 points
template<class T> T getSqDist(
        const std::array<T, 2> &p1,
        const std::array<T, 2> &p2) {

    auto dx = p1[0] - p2[0];
    auto dy = p1[1] - p2[1];
    return dx * dx + dy * dy;
}


// square distance from a point to a segment
template<class T> T sqSegDist(
        const std::array<T, 2> &p,
        const std::array<T, 2> &p1,
        const std::array<T, 2> &p2) {

    auto x = p1[0];
    auto y = p1[1];
    auto dx = p2[0] - x;
    auto dy = p2[1] - y;

    if (dx != 0 || dy != 0) {
        auto t = ((p[0] - x) * dx + (p[1] - y) * dy) / (dx * dx + dy * dy);
        if (t > 1) {
            x = p2[0];
            y = p2[1];
        } else if (t > 0) {
            x += dx * t;
            y += dy * t;
        }
    }

    dx = p[0] - x;
    dy = p[1] - y;

    return dx * dx + dy * dy;
}


// segment to segment distance, ported from http://geomalgorithms.com/a07-_distance.html by Dan Sunday
template<class T> T sqSegSegDist(T x0, T y0,
                                 T x1, T y1,
                                 T x2, T y2,
                                 T x3, T y3) {
    auto ux = x1 - x0;
    auto uy = y1 - y0;
    auto vx = x3 - x2;
    auto vy = y3 - y2;
    auto wx = x0 - x2;
    auto wy = y0 - y2;
    auto a = ux * ux + uy * uy;
    auto b = ux * vx + uy * vy;
    auto c = vx * vx + vy * vy;
    auto d = ux * wx + uy * wy;
    auto e = vx * wx + vy * wy;
    auto D = a * c - b * b;

    T sc, sN, tc, tN;
    auto sD = D;
    auto tD = D;

    if (D == 0) {
        sN = 0;
        sD = 1;
        tN = e;
        tD = c;
    } else {
        sN = b * e - c * d;
        tN = a * e - b * d;
        if (sN < 0) {
            sN = 0;
            tN = e;
            tD = c;
        } else if (sN > sD) {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0) {
        tN = 0;
        if (-d < 0) sN = 0;
        else if (-d > a) sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    } else if (tN > tD) {
        tN = tD;
        if (-d + b < 0) sN = 0;
        else if (-d + b > a) sN = sD;
        else {
            sN = -d + b;
            sD = a;
        }
    }

    sc = ((sN == 0) ? 0 : sN / sD);
    tc = ((tN == 0) ? 0 : tN / tD);

    auto cx = (1 - sc) * x0 + sc * x1;
    auto cy = (1 - sc) * y0 + sc * y1;
    auto cx2 = (1 - tc) * x2 + tc * x3;
    auto cy2 = (1 - tc) * y2 + tc * y3;
    auto dx = cx2 - cx;
    auto dy = cy2 - cy;

    return dx * dx + dy * dy;
}


template<class T, int DIM, int MAX_CHILDREN, class DATA> class rtree {
public:
    typedef rtree<T, DIM, MAX_CHILDREN, DATA> type;
    typedef const type const_type;
    typedef type *type_ptr;
    typedef const type *type_const_ptr;
    typedef std::array<T, DIM * 2> bounds_type;
    typedef DATA data_type;

    rtree():
            m_is_leaf(false), m_data() {
        for (auto i = 0; i < DIM; i++) {
            m_bounds[i] = std::numeric_limits<T>::max();
            m_bounds[i + DIM] = std::numeric_limits<T>::min();
        }
    }

    rtree(data_type data, const bounds_type &bounds):
            m_is_leaf(true), m_data(data), m_bounds(bounds) {
        for (auto i = 0; i < DIM; i++)
            if (bounds[i] > bounds[i + DIM])
                throw std::runtime_error("Bounds minima have to be less than maxima");
    }

    void insert(data_type data, const bounds_type &bounds) {
        if (m_is_leaf)
            throw std::runtime_error("Cannot insert into leaves");

        m_bounds = updated_bounds(bounds);
        if (m_children.size() < MAX_CHILDREN) {
            auto r = make_unique<type>(data, bounds);
            m_children.push_back(std::move(r));
            return;
        }

        std::reference_wrapper<type> best_child = *m_children.begin()->get();
        auto best_volume = volume(best_child.get().updated_bounds(bounds));
        for (auto it = ++m_children.begin(); it != m_children.end(); it++) {
            auto v = volume((*it)->updated_bounds(bounds));
            if (v < best_volume) {
                best_volume = v;
                best_child = *it->get();
            }
        }
        if (!best_child.get().is_leaf()) {
            best_child.get().insert(data, bounds);
#ifdef DEBUG
            std::cout << "best_child: " << bounds[0] << " " << bounds[1] << std::endl;
#endif
            return;
        }

        auto leaf = make_unique<type>(best_child.get().data(),
                                      best_child.get().bounds());
        best_child.get().m_is_leaf = false;
        best_child.get().m_data = data_type();
        best_child.get().m_children.push_back(std::move(leaf));
        best_child.get().insert(data, bounds);
    }

    void intersection(const bounds_type &bounds,
                      std::vector<std::reference_wrapper<const_type>> &res) const {
        if (!intersects(bounds))
            return;
        if (m_is_leaf) {
            res.push_back(*this);
            return;
        }
        for (auto &ch : m_children)
            ch->intersection(bounds, res);
    }

    std::vector<std::reference_wrapper<const_type>> intersection(const bounds_type& bounds) const {
        std::vector<std::reference_wrapper<const_type>> res;
        intersection(bounds, res);
        return res;
    }

    bool intersects(const bounds_type &bounds) const {
        for (auto i = 0; i < DIM; i++) {
            if (m_bounds[i] > bounds[i + DIM])
                return false;
            if (m_bounds[i + DIM] < bounds[i])
                return false;
        }
        return true;
    }

    void erase(data_type data, const bounds_type &bounds) {
        if (m_is_leaf)
            throw std::runtime_error("Cannot erase from leaves");

        if (!intersects(bounds))
            return;

        for (auto it = m_children.begin(); it != m_children.end(); ) {
            if (!(*it)->m_is_leaf) {
                (*it)->erase(data, bounds);
                it++;
            } else if ((*it)->m_data == data &&
                       (*it)->m_bounds == bounds) {
                m_children.erase(it++);
            } else
                it++;
        }
    }

    void print(int level = 0) {
        // print the entire tree

        for (auto it = m_children.begin(); it != m_children.end(); ) {
            auto bounds = (*it)->m_bounds;
            std::string pad(level, '\t');
            if ((*it)->m_is_leaf) {
                printf ("%s leaf %0.6f %0.6f \n", pad.c_str(), bounds[0], bounds[1]);
            }
            else {
                printf ("%s branch %0.6f %0.6f %0.6f %0.6f \n", pad.c_str(), bounds[0], bounds[1], bounds[2], bounds[3]);
                (*it)->print(level + 1);
            }
            it++;
        }
    }

    bounds_type updated_bounds(const bounds_type &child_bounds) const {
        bounds_type res;
        for (auto i = 0; i < DIM; i++) {
            res[i] = std::min(child_bounds[i], m_bounds[i]);
            res[i + DIM] = std::max(child_bounds[i + DIM], m_bounds[i + DIM]);
        }
        return res;
    }

    static T volume(const bounds_type &bounds) {
        T res = 1;
        for (auto i = 0; i < DIM; i++) {
            auto delta = bounds[i + DIM] - bounds[i];
            res *= delta;
        }
        return res;
    }

    const bounds_type& bounds() const {
        return m_bounds;
    }

    bool is_leaf() const {
        return m_is_leaf;
    }

    data_type data() const {
        return m_data;
    }

    const std::list<std::unique_ptr<type>>& children() const {
        return m_children;
    }

    static std::string bounds_to_string(const bounds_type &bounds) {
        std::string res = "( ";
        for (auto i = 0; i < DIM * 2; i++) {
            if (i > 0)
                res += ", ";
            res += std::to_string(bounds[i]);
        }
        res += " )";
        return res;
    }

    void to_string(std::string &res, int tab) const {
        std::string pad(tab, '\t');

        if (m_is_leaf) {
            res += pad + "{ data: " + std::to_string(m_data) +
                   ", bounds: " + bounds_to_string(m_bounds) +
                   " }";
            return;
        }

        res += pad + "{ bounds: " + bounds_to_string(m_bounds) +
               ", children: [\n";
        auto i = 0;
        for (auto &ch : m_children) {
            if (i++ > 0)
                res += "\n";
            ch->to_string(res, tab + 1);
        }
        res += "\n" + pad + "]}";
    }

    std::string to_string() const {
        std::string res;
        to_string(res, 0);
        return res;
    }

private:
    bool m_is_leaf;
    data_type m_data;
    std::list<std::unique_ptr<type>> m_children;
    bounds_type m_bounds;
};


template<class T> struct Node {
    typedef Node<T> type;
    typedef type *type_ptr;
    typedef std::array<T, 2> point_type;

    Node(): p(),
            minX(), minY(), maxX(), maxY() {

    }

    Node(const point_type &p): Node() {
        this->p = p;
    }

    point_type p;
    T minX;
    T minY;
    T maxX;
    T maxY;
};


template <class T> class CircularList;


template<class T> class CircularElement {
public:
    typedef CircularElement<T> type;
    typedef type *ptr_type;

    template<class... Args> CircularElement<T>(Args&&... args):
            m_data(std::forward<Args>(args)...) {

    }

    T& data() {
        return m_data;
    }

    template<class... Args> CircularElement<T>* insert(Args&&... args) {
        auto elem = new CircularElement<T>(std::forward<Args>(args)...);
        elem->m_prev = this;
        elem->m_next = m_next;
        m_next->m_prev = elem;
        m_next = elem;
        return elem;
    }

    CircularElement<T>* prev() {
        return m_prev;
    }

    CircularElement<T>* next() {
        return m_next;
    }

private:
    T m_data;
    CircularElement<T> *m_prev;
    CircularElement<T> *m_next;

    friend class CircularList<T>;
};


template<class T> class CircularList {
public:
    typedef CircularElement<T> element_type;

    CircularList(): m_last(nullptr) {

    }

    ~CircularList() {
#ifdef DEBUG
        std::cout << "~CircularList()" << std::endl;
#endif
        auto node = m_last;
        while (true) {
#ifdef DEBUG
//             std::cout << (i++) << std::endl;
#endif
            auto tmp = node;
            node = node->m_next;
            delete tmp;
            if (node == m_last)
                break;
        }
    }

    template<class... Args> CircularElement<T>* insert(element_type *prev, Args&&... args) {
        auto elem = new CircularElement<T>(std::forward<Args>(args)...);

        if (prev == nullptr && m_last != nullptr)
            throw std::runtime_error("Once the list is non-empty you must specify where to insert");

        if (prev == nullptr) {
            elem->m_prev = elem->m_next = elem;
        } else {
            elem->m_prev = prev;
            elem->m_next = prev->m_next;
            prev->m_next->m_prev = elem;
            prev->m_next = elem;
        }

        m_last = elem;

        return elem;
    }


private:
    element_type *m_last;
};


// update the bounding box of a node's edge
template<class T> void updateBBox(typename CircularElement<T>::ptr_type elem) {
    auto &node(elem->data());
    auto p1 = node.p;
    auto p2 = elem->next()->data().p;
    node.minX = std::min(p1[0], p2[0]);
    node.minY = std::min(p1[1], p2[1]);
    node.maxX = std::max(p1[0], p2[0]);
    node.maxY = std::max(p1[1], p2[1]);
}


#ifdef DEBUG_2
template<class T> void snapshot(
    const std::array<T, 2> &a,
    const std::array<T, 2> &b,
    const std::array<T, 2> &c,
    const std::array<T, 2> &d,
    const double sqLen,
    const double maxSqLen,
    const std::array<T, 2> &trigger,
    const bool use_trigger) {

    if ( !use_trigger || trigger == b ) {
        if ( !use_trigger )
            printf ("Snapshot untriggered\n");
        else
            printf ("Snapshot trigger: %0.6f %0.6f \n", trigger[0], trigger[1]);
        printf ("... segment a, b: %0.6f %0.6f, %0.6f %0.6f \n", a[0], a[1], b[0], b[1]);
        printf ("... segment c, d: %0.6f %0.6f, %0.6f %0.6f \n", c[0], c[1], d[0], d[1]);
        printf ("... sqDist a-b, b-c, c-d: %e, %e, %e", getSqDist(a, b), getSqDist(b, c), getSqDist(c, d));
        printf ("... sqLen, maxSqLen: %e, %e", sqLen, maxSqLen);
    }
}
#endif


template<class T, int MAX_CHILDREN> std::vector<std::array<T, 2>> concaveman(
        const std::vector<std::array<T, 2>> &points,
        // start with a convex hull of the points
        const std::vector<int> &hull,
        // a relative measure of concavity; higher value means simpler hull
        T concavity=2,
        // when a segment goes below this length threshold, it won't be drilled down further
        T lengthThreshold=0
) {

    typedef Node<T> node_type;
    typedef std::array<T, 2> point_type;
    typedef CircularElement<node_type> circ_elem_type;
    typedef CircularList<node_type> circ_list_type;
    typedef circ_elem_type *circ_elem_ptr_type;

#ifdef DEBUG
    std::cout << "concaveman()" << std::endl;
#endif

    // exit if hull includes all points already
    if (hull.size() == points.size()) {
        std::vector<point_type> res;
        for (auto &i : hull) res.push_back(points[i]);
        return res;
    }

    // index the points with an R-tree
    rtree<T, 2, MAX_CHILDREN, point_type> tree;

    for (auto &p : points)
        tree.insert(p, { p[0], p[1], p[0], p[1] });

    circ_list_type circList;
    circ_elem_ptr_type last = nullptr;

    std::list<circ_elem_ptr_type> queue;

    // turn the convex hull into a linked list and populate the initial edge queue with the nodes
    for (auto &idx : hull) {
        auto &p = points[idx];
        tree.erase(p, { p[0], p[1], p[0], p[1] });
        last = circList.insert(last, p);
        queue.push_back(last);
    }

#ifdef DEBUG_2
    tree.print(0);
#endif

    // loops through the hull?  why?
#ifdef DEBUG
    std::cout << "Starting hull: ";
#endif
    for (auto elem = last->next(); ; elem=elem->next()) {
#ifdef DEBUG
        std::cout << elem->data().p[0] << " " << elem->data().p[1] << std::endl;
#endif
        if (elem == last)
            break;
    }

    // index the segments with an R-tree (for intersection checks)
    rtree<T, 2, MAX_CHILDREN, circ_elem_ptr_type> segTree;
    for (auto &elem : queue) {
        auto &node(elem->data());
        updateBBox<node_type>(elem);
        segTree.insert(elem, { node.minX,
                               node.minY, node.maxX, node.maxY });
    }

    auto sqConcavity = concavity * concavity;
    auto sqLenThreshold = lengthThreshold * lengthThreshold;

    // process edges one by one
    while (!queue.empty()) {
        auto elem = *queue.begin();
        queue.pop_front();

        auto a = elem->prev()->data().p;
        auto b = elem->data().p;
        auto c = elem->next()->data().p;
        auto d = elem->next()->next()->data().p;

        // skip the edge if it's already short enough
        auto sqLen = getSqDist(b, c);
        if (sqLen < sqLenThreshold)
            continue;

        auto maxSqLen = sqLen / sqConcavity;

#ifdef DEBUG_2
        // dump key parameters either on every pass or when a certain point is 'b'
        point_type trigger = { 151.1373474787800, -33.7733192376544 };
        snapshot(a, b, c, d, sqLen, maxSqLen, trigger, true);
#endif

        // find the best connection point for the current edge to flex inward to
        bool ok;
        auto p = findCandidate(tree, a, b, c, d, maxSqLen, segTree, ok);

        // if we found a connection and it satisfies our concavity measure
        if (ok && std::min(getSqDist(p, b), getSqDist(p, c)) <= maxSqLen) {

#ifdef DEBUG
            printf ("Modifying hull, p: %0.6f %0.6f \n" ,p[0], p[1]);
#endif

            // connect the edge endpoints through this point and add 2 new edges to the queue
            queue.push_back(elem);
            queue.push_back(elem->insert(p));

            // update point and segment indexes
            auto &node = elem->data();
            auto &next = elem->next()->data();

            tree.erase(p, { p[0], p[1], p[0], p[1] });
            segTree.erase(elem, { node.minX, node.minY, node.maxX, node.maxY });

            updateBBox<node_type>(elem);
            updateBBox<node_type>(elem->next());

            segTree.insert(elem, { node.minX, node.minY, node.maxX, node.maxY });
            segTree.insert(elem->next(), { next.minX, next.minY, next.maxX, next.maxY });
        }
#ifdef DEBUG
        else
            printf ("No point found along segment: %0.6f %0.6f, %0.6f %0.6f \n", b[0], b[1], c[0], c[1]);
#endif
    }

    // convert the resulting hull linked list to an array of points
    std::vector<point_type> concave;
    for (auto elem = last->next(); ; elem = elem->next()) {
        concave.push_back(elem->data().p);
        if (elem == last)
            break;
    }

    return concave;
}


template<class T, int MAX_CHILDREN> std::array<T, 2> findCandidate(
        const rtree<T, 2, MAX_CHILDREN, std::array<T, 2>> &tree,
const std::array<T, 2> &a,
const std::array<T, 2> &b,
const std::array<T, 2> &c,
const std::array<T, 2> &d,
        T maxDist,
const rtree<T, 2, MAX_CHILDREN, typename CircularElement<Node<T>>::ptr_type> &segTree,
bool &ok) {

typedef std::array<T, 2> point_type;
typedef CircularElement<Node<T>> circ_elem_type;
typedef rtree<T, 2, MAX_CHILDREN, std::array<T, 2>> tree_type;
typedef const tree_type const_tree_type;
typedef std::reference_wrapper<const_tree_type> tree_ref_type;
typedef std::tuple<T, tree_ref_type> tuple_type;

#ifdef DEBUG
std::cout << "findCandidate(), maxDist: " << maxDist << std::endl;
#endif

ok = false;

std::priority_queue<tuple_type, std::vector<tuple_type>, compare_first<tuple_type>> queue;
std::reference_wrapper<const_tree_type> node = tree;

// search through the point R-tree with a depth-first search using a priority queue
// in the order of distance to the edge (b, c)
while (true) {
for (auto &child : node.get().children()) {

auto bounds = child->bounds();
point_type pt = { bounds[0], bounds[1] };

auto dist = child->is_leaf() ? sqSegDist(pt, b, c) : sqSegBoxDist(b, c, *child);
if (dist > maxDist)
continue;  // skip the node if it's farther than we ever need

queue.push(tuple_type(-dist, *child));
}

while (!queue.empty() && std::get<1>(queue.top()).get().is_leaf()) {
auto item = queue.top();
queue.pop();

auto bounds = std::get<1>(item).get().bounds();
point_type p = { bounds[0], bounds[1] };

// skip all points that are as close to adjacent edges (a,b) and (c,d),
// and points that would introduce self-intersections when connected
auto d0 = sqSegDist(p, a, b);
auto d1 = sqSegDist(p, c, d);

#ifdef DEBUG_2
printf ("    p: %0.6f %0.6f sqSegDist: %e, %e, %e \n", bounds[0], bounds[1], d0, std::get<0>(item), d1);
#endif

if (-std::get<0>(item) < d0 && -std::get<0>(item) < d1 &&
        noIntersections(b, p, segTree) &&
noIntersections(c, p, segTree)) {

ok = true;
return std::get<1>(item).get().data();
}

#ifdef DEBUG_2
else {
                bool cond1 = -std::get<0>(item) < d0;
                bool cond2 = -std::get<0>(item) < d1;
                bool cond3 = noIntersections(b, p, segTree);
                bool cond4 = noIntersections(c, p, segTree);
                std::cout << "Not OK: " << cond1 << " " << cond2 << " " << cond3 << " " << cond4 << std::endl;
            }
#endif
}

if (queue.empty())
break;

node = std::get<1>(queue.top());
queue.pop();
}

return point_type();
}


// square distance from a segment bounding box to the given one
template<class T, int MAX_CHILDREN, class USER_DATA> T sqSegBoxDist(
        const std::array<T, 2> &a,
        const std::array<T, 2> &b,
        const rtree<T, 2, MAX_CHILDREN, USER_DATA> &bbox) {

    if (inside(a, bbox) || inside(b, bbox))
        return 0;

    auto &bounds = bbox.bounds();
    auto minX = bounds[0];
    auto minY = bounds[1];
    auto maxX = bounds[2];
    auto maxY = bounds[3];

    auto d1 = sqSegSegDist(a[0], a[1], b[0], b[1], minX, minY, maxX, minY);
    if (d1 == 0) return 0;

    auto d2 = sqSegSegDist(a[0], a[1], b[0], b[1], minX, minY, minX, maxY);
    if (d2 == 0) return 0;

    auto d3 = sqSegSegDist(a[0], a[1], b[0], b[1], maxX, minY, maxX, maxY);
    if (d3 == 0) return 0;

    auto d4 = sqSegSegDist(a[0], a[1], b[0], b[1], minX, maxY, maxX, maxY);
    if (d4 == 0) return 0;

    return std::min(std::min(d1, d2), std::min(d3, d4));
}


template<class T, int MAX_CHILDREN, class USER_DATA> bool inside(
        const std::array<T, 2> &a,
        const rtree<T, 2, MAX_CHILDREN, USER_DATA> &bbox) {

    auto &bounds = bbox.bounds();

    auto minX = bounds[0];
    auto minY = bounds[1];
    auto maxX = bounds[2];
    auto maxY = bounds[3];

    auto res = (a[0] >= minX) &&
               (a[0] <= maxX) &&
               (a[1] >= minY) &&
               (a[1] <= maxY);
    return res;
}


// check if the edge (a,b) doesn't intersect any other edges
template<class T, int MAX_CHILDREN> bool noIntersections(
        const std::array<T, 2> &a,
        const std::array<T, 2> &b,
        const rtree<T, 2, MAX_CHILDREN, typename CircularElement<Node<T>>::ptr_type> &segTree) {

    auto minX = std::min(a[0], b[0]);
    auto minY = std::min(a[1], b[1]);
    auto maxX = std::max(a[0], b[0]);
    auto maxY = std::max(a[1], b[1]);

    auto isect = segTree.intersection({ minX, minY, maxX, maxY });

    for (decltype(segTree) &ch : isect) {
        auto elem = ch.data();

        if (intersects(elem->data().p, elem->next()->data().p, a, b))
            return false;
    }

    return true;
}

#endif //OPENCV_CONCAVEMAN_HPP
