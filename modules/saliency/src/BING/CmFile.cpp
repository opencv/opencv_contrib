/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "CmFile.h"

// Get image names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
int CmFile::GetNames( CStr &_nameW, vecS &_names, std::string _dir )
{
  _dir = GetFolder( _nameW );
  _names.clear();

  DIR *dir;
  struct dirent *ent;
  if( ( dir = opendir( _dir.c_str() ) ) != NULL )
  {
    //print all the files and directories within directory
    while ( ( ent = readdir( dir ) ) != NULL )
    {
      if( ent->d_name[0] == '.' )
        continue;
      if( ent->d_type == 4 )
        continue;
      _names.push_back( ent->d_name );
    }
    closedir( dir );
  }
  else
  {
    perror( "" );
    return EXIT_FAILURE;
  }
  return (int) _names.size();
}
int CmFile::GetSubFolders( CStr &folder, vecS &subFolders )
{
  subFolders.clear();
  std::string nameWC = GetFolder( folder );  //folder + "/*";

  DIR *dir;
  struct dirent *ent;
  if( ( dir = opendir( nameWC.c_str() ) ) != NULL )
  {
    while ( ( ent = readdir( dir ) ) != NULL )
    {
      if( ent->d_name[0] == '.' )
        continue;
      if( ent->d_type == 4 )
      {
        subFolders.push_back( ent->d_name );
      }
    }
    closedir( dir );
  }
  else
  {
    perror( "" );
    return EXIT_FAILURE;
  }
  return (int) subFolders.size();
}
int CmFile::GetNames( CStr& rootFolder, CStr &fileW, vecS &names )
{
  GetNames( rootFolder + fileW, names );
  vecS subFolders, tmpNames;
  int subNum = CmFile::GetSubFolders( rootFolder, subFolders );  //
  for ( int i = 0; i < subNum; i++ )
  {
    subFolders[i] += "/";
    int subNum_ = GetNames( rootFolder + subFolders[i], fileW, tmpNames );
    for ( int j = 0; j < subNum_; j++ )
      names.push_back( subFolders[i] + tmpNames[j] );
  }
  return (int) names.size();
}
int CmFile::GetNamesNE( CStr& nameWC, vecS &names, std::string dir, std::string ext )
{
  int fNum = GetNames( nameWC, names, dir );
  ext = GetExtention( nameWC );
  for ( int i = 0; i < fNum; i++ )
    names[i] = GetNameNE( names[i] );
  return fNum;
}
int CmFile::GetNamesNE( CStr& rootFolder, CStr &fileW, vecS &names )
{
  int fNum = GetNames( rootFolder, fileW, names );
  int extS = GetExtention( fileW ).size();
  for ( int i = 0; i < fNum; i++ )
    names[i].resize( names[i].size() - extS );
  return fNum;
}
bool CmFile::MkDir( CStr &_path )
{
  if( _path.size() == 0 )
    return false;
  static char buffer[1024];
  strcpy( buffer, _S( _path ) );
#ifdef _WIN32
  for (int i = 0; buffer[i] != 0; i ++)
  {
    if (buffer[i] == '\\' || buffer[i] == '/')
    {
      buffer[i] = '\0';
      CreateDirectoryA(buffer, 0);
      buffer[i] = '/';
    }
  }
  return CreateDirectoryA(_S(_path), 0);
#else
  for ( int i = 0; buffer[i] != 0; i++ )
  {
    if( buffer[i] == '\\' || buffer[i] == '/' )
    {
      buffer[i] = '\0';
      mkdir( buffer, 0 );
      buffer[i] = '/';
    }
  }
  return mkdir( _S( _path ), 0 );
#endif
}
void CmFile::loadStrList( CStr &fName, vecS & strs, bool flag )
{
  std::ifstream fIn( fName.c_str() );
  std::string line;
  //vecS strs;
  while ( getline( fIn, line ) )
  {
    unsigned sz = line.size();
    if( flag )
      line.resize( sz - 1 );
    printf( "%s\n", _S( line ) );
    strs.push_back( line );
  }
  //return strs;
}
bool CmFile::writeStrList( CStr &fName, const vecS &strs )
{
  FILE *f = fopen( _S( fName ), "w" );
  if( f == NULL )
    return false;
  for ( size_t i = 0; i < strs.size(); i++ )
    fprintf( f, "%s\n", _S( strs[i] ) );
  fclose( f );
  return true;
}
