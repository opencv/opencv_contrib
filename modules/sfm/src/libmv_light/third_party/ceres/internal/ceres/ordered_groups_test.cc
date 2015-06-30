// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/ordered_groups.h"

#include <cstddef>
#include <vector>
#include "gtest/gtest.h"
#include "ceres/collections_port.h"

namespace ceres {
namespace internal {

TEST(OrderedGroup, EmptyOrderedGroupBehavesCorrectly) {
  ParameterBlockOrdering ordering;
  EXPECT_EQ(ordering.NumGroups(), 0);
  EXPECT_EQ(ordering.NumElements(), 0);
  EXPECT_EQ(ordering.GroupSize(1), 0);
  double x;
  EXPECT_EQ(ordering.GroupId(&x), -1);
  EXPECT_FALSE(ordering.Remove(&x));
}

TEST(OrderedGroup, EverythingInOneGroup) {
  ParameterBlockOrdering ordering;
  double x[3];
  ordering.AddElementToGroup(x, 1);
  ordering.AddElementToGroup(x + 1, 1);
  ordering.AddElementToGroup(x + 2, 1);
  ordering.AddElementToGroup(x, 1);

  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumElements(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);
  EXPECT_EQ(ordering.GroupId(x), 1);
  EXPECT_EQ(ordering.GroupId(x + 1), 1);
  EXPECT_EQ(ordering.GroupId(x + 2), 1);

  ordering.Remove(x);
  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumElements(), 2);
  EXPECT_EQ(ordering.GroupSize(1), 2);
  EXPECT_EQ(ordering.GroupSize(0), 0);

  EXPECT_EQ(ordering.GroupId(x), -1);
  EXPECT_EQ(ordering.GroupId(x + 1), 1);
  EXPECT_EQ(ordering.GroupId(x + 2), 1);
}

TEST(OrderedGroup, StartInOneGroupAndThenSplit) {
  ParameterBlockOrdering ordering;
  double x[3];
  ordering.AddElementToGroup(x, 1);
  ordering.AddElementToGroup(x + 1, 1);
  ordering.AddElementToGroup(x + 2, 1);
  ordering.AddElementToGroup(x, 1);

  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumElements(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);
  EXPECT_EQ(ordering.GroupId(x), 1);
  EXPECT_EQ(ordering.GroupId(x + 1), 1);
  EXPECT_EQ(ordering.GroupId(x + 2), 1);

  ordering.AddElementToGroup(x, 5);
  EXPECT_EQ(ordering.NumGroups(), 2);
  EXPECT_EQ(ordering.NumElements(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 2);
  EXPECT_EQ(ordering.GroupSize(5), 1);
  EXPECT_EQ(ordering.GroupSize(0), 0);

  EXPECT_EQ(ordering.GroupId(x), 5);
  EXPECT_EQ(ordering.GroupId(x + 1), 1);
  EXPECT_EQ(ordering.GroupId(x + 2), 1);
}

TEST(OrderedGroup, AddAndRemoveEveryThingFromOneGroup) {
  ParameterBlockOrdering ordering;
  double x[3];
  ordering.AddElementToGroup(x, 1);
  ordering.AddElementToGroup(x + 1, 1);
  ordering.AddElementToGroup(x + 2, 1);
  ordering.AddElementToGroup(x, 1);

  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumElements(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);
  EXPECT_EQ(ordering.GroupId(x), 1);
  EXPECT_EQ(ordering.GroupId(x + 1), 1);
  EXPECT_EQ(ordering.GroupId(x + 2), 1);

  ordering.AddElementToGroup(x, 5);
  ordering.AddElementToGroup(x + 1, 5);
  ordering.AddElementToGroup(x + 2, 5);
  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumElements(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 0);
  EXPECT_EQ(ordering.GroupSize(5), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);

  EXPECT_EQ(ordering.GroupId(x), 5);
  EXPECT_EQ(ordering.GroupId(x + 1), 5);
  EXPECT_EQ(ordering.GroupId(x + 2), 5);
}

TEST(OrderedGroup, ReverseOrdering) {
  ParameterBlockOrdering ordering;
  double x[3];
  ordering.AddElementToGroup(x, 1);
  ordering.AddElementToGroup(x + 1, 2);
  ordering.AddElementToGroup(x + 2, 2);

  EXPECT_EQ(ordering.NumGroups(), 2);
  EXPECT_EQ(ordering.NumElements(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 1);
  EXPECT_EQ(ordering.GroupSize(2), 2);
  EXPECT_EQ(ordering.GroupId(x), 1);
  EXPECT_EQ(ordering.GroupId(x + 1), 2);
  EXPECT_EQ(ordering.GroupId(x + 2), 2);

  ordering.Reverse();

  EXPECT_EQ(ordering.NumGroups(), 2);
  EXPECT_EQ(ordering.NumElements(), 3);
  EXPECT_EQ(ordering.GroupSize(3), 1);
  EXPECT_EQ(ordering.GroupSize(2), 2);
  EXPECT_EQ(ordering.GroupId(x), 3);
  EXPECT_EQ(ordering.GroupId(x + 1), 2);
  EXPECT_EQ(ordering.GroupId(x + 2), 2);
}

}  // namespace internal
}  // namespace ceres
