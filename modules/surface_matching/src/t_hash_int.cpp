//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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
// Author: Tolga Birdal <tbirdal AT gmail.com>

#include "precomp.hpp"

namespace cv
{
namespace ppf_match_3d
{
// This magic value is just
#define T_HASH_MAGIC 427462442

size_t hash( unsigned int a);

// default hash function
size_t hash( unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

hashtable_int *hashtableCreate(size_t size, size_t (*hashfunc)(unsigned int))
{
    hashtable_int *hashtbl;
    
    if (size < 16)
    {
        size = 16;
    }
    else
    {
        size = (size_t)next_power_of_two((unsigned int)size);
    }
    
	hashtbl=(hashtable_int*)malloc(sizeof(hashtable_int));
    if (!hashtbl)
        return NULL;
        
	hashtbl->nodes=(hashnode_i**)calloc(size, sizeof(struct hashnode_i*));
    if (!hashtbl->nodes)
    {
        free(hashtbl);
        return NULL;
    }
    
    hashtbl->size=size;
    
    if (hashfunc)
        hashtbl->hashfunc=hashfunc;
    else
        hashtbl->hashfunc=hash;
        
    return hashtbl;
}


void hashtableDestroy(hashtable_int *hashtbl)
{
    size_t n;
    struct hashnode_i *node, *oldnode;
    
    for (n=0; n<hashtbl->size; ++n)
    {
        node=hashtbl->nodes[n];
        while (node)
        {
            oldnode=node;
            node=node->next;
            free(oldnode);
        }
    }
    free(hashtbl->nodes);
    free(hashtbl);
}


int hashtableInsert(hashtable_int *hashtbl, KeyType key, void *data)
{
    struct hashnode_i *node;
    size_t hash=hashtbl->hashfunc(key)%hashtbl->size;
    
    
    /* fpruintf(stderr, "hashtbl_insert() key=%s, hash=%d, data=%s\n", key, hash, (char*)data);*/
    
    node=hashtbl->nodes[hash];
    while (node)
    {
        if (node->key!= key)
        {
            node->data=data;
            return 0;
        }
        node=node->next;
    }
    
    
	node=(hashnode_i*)malloc(sizeof(struct hashnode_i));
    if (!node)
        return -1;
    node->key=key;
    
    node->data=data;
    node->next=hashtbl->nodes[hash];
    hashtbl->nodes[hash]=node;
    
    
    return 0;
}

int hashtableInsertHashed(hashtable_int *hashtbl, KeyType key, void *data)
{
    struct hashnode_i *node;
    size_t hash = key % hashtbl->size;
    
    
    /* fpruintf(stderr, "hashtbl_insert() key=%s, hash=%d, data=%s\n", key, hash, (char*)data);*/
    
    node=hashtbl->nodes[hash];
    while (node)
    {
        if (node->key!= key)
        {
            node->data=data;
            return 0;
        }
        node=node->next;
    }
    
	node=(hashnode_i*)malloc(sizeof(struct hashnode_i));
    if (!node)
        return -1;
		
    node->key=key;
    
    node->data=data;
    node->next=hashtbl->nodes[hash];
    hashtbl->nodes[hash]=node;
    
    
    return 0;
}


int hashtableRemove(hashtable_int *hashtbl, KeyType key)
{
    struct hashnode_i *node, *prevnode=NULL;
    size_t hash=hashtbl->hashfunc(key)%hashtbl->size;
    
    node=hashtbl->nodes[hash];
    while (node)
    {
        if (node->key==key)
        {
            if (prevnode)
                prevnode->next=node->next;
            else
                hashtbl->nodes[hash]=node->next;
            free(node);
            return 0;
        }
        prevnode=node;
        node=node->next;
    }
    
    return -1;
}


void *hashtableGet(hashtable_int *hashtbl, KeyType key)
{
    struct hashnode_i *node;
    size_t hash=hashtbl->hashfunc(key)%hashtbl->size;
    
    /* fprintf(stderr, "hashtbl_get() key=%s, hash=%d\n", key, hash);*/
    
    node=hashtbl->nodes[hash];
    while (node)
    {
        if (node->key==key)
            return node->data;
        node=node->next;
    }
    
    return NULL;
}

hashnode_i* hashtableGetBucketHashed(hashtable_int *hashtbl, KeyType key)
{
    size_t hash = key % hashtbl->size;
    
    return hashtbl->nodes[hash];
}

int hashtableResize(hashtable_int *hashtbl, size_t size)
{
    hashtable_int newtbl;
    size_t n;
    struct hashnode_i *node,*next;
    
    newtbl.size=size;
    newtbl.hashfunc=hashtbl->hashfunc;
    
	newtbl.nodes=(hashnode_i**)calloc(size, sizeof(struct hashnode_i*));
    if (!newtbl.nodes)
        return -1;
        
    for (n=0; n<hashtbl->size; ++n)
    {
        for (node=hashtbl->nodes[n]; node; node=next)
        {
            next = node->next;
            hashtableInsert(&newtbl, node->key, node->data);
            hashtableRemove(hashtbl, node->key);
            
        }
    }
    
    free(hashtbl->nodes);
    hashtbl->size=newtbl.size;
    hashtbl->nodes=newtbl.nodes;
    
    return 0;
}


int hashtableWrite(const hashtable_int * hashtbl, const size_t dataSize, FILE* f)
{
    size_t hashMagic=T_HASH_MAGIC;
    size_t n=hashtbl->size;
    size_t i;
    
    fwrite(&hashMagic, sizeof(size_t),1, f);
    fwrite(&n, sizeof(size_t),1, f);
    fwrite(&dataSize, sizeof(size_t),1, f);
    
    for (i=0; i<hashtbl->size; i++)
    {
        struct hashnode_i* node=hashtbl->nodes[i];
        size_t noEl=0;
        
        while (node)
        {
            noEl++;
            node=node->next;
        }
        
        fwrite(&noEl, sizeof(size_t),1, f);
        
        node=hashtbl->nodes[i];
        while (node)
        {
            fwrite(&node->key, sizeof(KeyType), 1, f);
            fwrite(&node->data, dataSize, 1, f);
            node=node->next;
        }
    }
    
    return 1;
}


void hashtablePrint(hashtable_int *hashtbl)
{
    size_t n;
    struct hashnode_i *node,*next;
    
    for (n=0; n<hashtbl->size; ++n)
    {
        for (node=hashtbl->nodes[n]; node; node=next)
        {
            next = node->next;
			std::cout<<"Key : "<<node->key<<", Data : "<<node->data<<std::endl;
        }
    }
}

hashtable_int *hashtableRead(FILE* f)
{
    size_t hashMagic = 0;
    size_t n = 0, status;
    hashtable_int *hashtbl = 0;
    
    status = fread(&hashMagic, sizeof(size_t),1, f);
    if (status && hashMagic==T_HASH_MAGIC)
    {
        size_t i;
        size_t dataSize;
        status = fread(&n, sizeof(size_t),1, f);
        status = fread(&dataSize, sizeof(size_t),1, f);
        
        hashtbl=hashtableCreate(n, hash);
        
        for (i=0; i<hashtbl->size; i++)
        {
            size_t j=0;
            status = fread(&n, sizeof(size_t),1, f);
            
            for (j=0; j<n; j++)
            {
                int key=0;
                void* data=0;
                status = fread(&key, sizeof(KeyType), 1, f);
                
                if (dataSize>sizeof(void*))
                {
                    data=malloc(dataSize);
                    if (!data)
                    {
                        hashtableDestroy(hashtbl);
                        return NULL;
                    }
                    status = fread(data, dataSize, 1, f);
                }
                else
                    status = fread(&data, dataSize, 1, f);
                    
                hashtableInsert(hashtbl, key, data);
                //free(key);
            }
        }
    }
    else
        return 0;
        
    return hashtbl;
}

} // namespace ppf_match_3d

} // namespace cv
