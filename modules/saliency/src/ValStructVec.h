#pragma once

/************************************************************************/
/* A value struct vector that supports efficient sorting                */
/************************************************************************/

template<typename VT, typename ST>
struct ValStructVec
{
  ValStructVec()
  {
    clear();
  }
  inline int size() const
  {
    return sz;
  }
  inline void clear()
  {
    sz = 0;
    structVals.clear();
    valIdxes.clear();
  }
  inline void reserve( int resSz )
  {
    clear();
    structVals.reserve( resSz );
    valIdxes.reserve( resSz );
  }
  inline void pushBack( const VT& val, const ST& structVal )
  {
    valIdxes.push_back( make_pair( val, sz ) );
    structVals.push_back( structVal );
    sz++;
  }

  inline const VT& operator ()( int i ) const
  {
    return valIdxes[i].first;
  }  // Should be called after sort
  inline const ST& operator []( int i ) const
  {
    return structVals[valIdxes[i].second];
  }  // Should be called after sort
  inline VT& operator ()( int i )
  {
    return valIdxes[i].first;
  }  // Should be called after sort
  inline ST& operator []( int i )
  {
    return structVals[valIdxes[i].second];
  }  // Should be called after sort

  void sort( bool descendOrder = true );
  const vector<ST> &getSortedStructVal();
  vector<pair<VT, int> > getvalIdxes();
  void append( const ValStructVec<VT, ST> &newVals, int startV = 0 );

  vector<ST> structVals;  // struct values

 private:
  int sz;  // size of the value struct vector
  vector<pair<VT, int> > valIdxes;  // Indexes after sort
  bool smaller()
  {
    return true;
  }
  ;
  vector<ST> sortedStructVals;
};

template<typename VT, typename ST>
void ValStructVec<VT, ST>::append( const ValStructVec<VT, ST> &newVals, int startV )
{
  int newValsSize = newVals.size();
  for ( int i = 0; i < newValsSize; i++ )
    pushBack( (float) ( ( i + 300 ) * startV )/*newVals(i)*/, newVals[i] );
}

template<typename VT, typename ST>
void ValStructVec<VT, ST>::sort( bool descendOrder /* = true */)
{
  if( descendOrder )
    std::sort( valIdxes.begin(), valIdxes.end(), std::greater<pair<VT, int> >() );
  else
    std::sort( valIdxes.begin(), valIdxes.end(), std::less<pair<VT, int> >() );
}

template<typename VT, typename ST>
const vector<ST>& ValStructVec<VT, ST>::getSortedStructVal()
{
  sortedStructVals.resize( sz );
  for ( int i = 0; i < sz; i++ )
    sortedStructVals[i] = structVals[valIdxes[i].second];
  return sortedStructVals;
}

template<typename VT, typename ST>
vector<pair<VT, int> > ValStructVec<VT, ST>::getvalIdxes()
{
  return valIdxes;
}

/*
 void valStructVecDemo()
 {
 ValStructVec<int, string> sVals;
 sVals.pushBack(3, "String 3");
 sVals.pushBack(5, "String 5");
 sVals.pushBack(4, "String 4");
 sVals.pushBack(1, "String 1");
 sVals.sort(false);
 for (int i = 0; i < sVals.size(); i++)
 printf("%d, %s\n", sVals(i), _S(sVals[i]));
 }
 */
