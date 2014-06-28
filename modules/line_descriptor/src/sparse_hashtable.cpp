#include "precomp.hpp"

const int SparseHashtable::MAX_B = 37;

/* constructor */
SparseHashtable::SparseHashtable()
{
    table = NULL;
    size = 0;
    b = 0;
}

/* initializer */
int SparseHashtable::init(int _b)
{
    b = _b;
    
    if (b < 5 || b > MAX_B || b > sizeof(UINT64)*8)
	return 1;
    
    size = UINT64_1 << (b-5);	// size = 2 ^ b
    table = (BucketGroup*) calloc(size, sizeof(BucketGroup));

    return 0;
}

/* destructor */
SparseHashtable::~SparseHashtable () {
    free(table);
}

/* insert data */
void SparseHashtable::insert(UINT64 index, UINT32 data) {
    table[index >> 5].insert((int)(index % 32), data);
}

/* query data */
UINT32* SparseHashtable::query(UINT64 index, int *Size) {
    return table[index >> 5].query((int)(index % 32), Size);
}
