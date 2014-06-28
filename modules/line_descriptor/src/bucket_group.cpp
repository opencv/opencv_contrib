#include "precomp.hpp"

/* constructor */
BucketGroup::BucketGroup()
{
    empty = 0;
    group = NULL;
}

/* destructor */
BucketGroup::~BucketGroup()
{
    if (group != NULL)
        delete group;
}

/* insert data into the bucket */
void BucketGroup::insert(int subindex, UINT32 data)
{
    if (group == NULL)
    {
        group = new Array32();
        group->push(0);
    }

    UINT32 lowerbits = ((UINT32)1 << subindex) - 1;
    int end = popcnt(empty & lowerbits);

    if (!(empty & ((UINT32)1 << subindex)))
    {
        group->insert(end, group->arr[end+2]);
        empty |= (UINT32)1 << subindex;
    }

    int totones = popcnt(empty);
    group->insert(totones+1+group->arr[2+end+1], data);
    for (int i=end+1; i<totones+1; i++)
        group->arr[2+i]++;
}

/* perform a query to the bucket */
UINT32* BucketGroup::query(int subindex, int *size)
{
    if (empty & ((UINT32)1 << subindex))
    {
        UINT32 lowerbits = ((UINT32)1 << subindex) - 1;
        int end = popcnt(empty & lowerbits);
        int totones = popcnt(empty);
        *size = group->arr[2+end+1]-group->arr[2+end];
        return group->arr + 2 + totones+1 + group->arr[2+end];
    }

    else
    {
        *size = 0;
        return NULL;
    }
}
