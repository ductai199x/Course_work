// THIS FILE IS JUST REFERENCE AND NOT USED IN THE PROJECT.
#ifndef __HASH_TABLE_HH__
#define __HASH_TABLE_HH__

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

typedef struct hash_node hash_node;

struct hash_node
{
    int key;
    uint64_t last_exe_cycle;
    uint64_t stall_cycles;
    hash_node *next;
};

typedef struct hash_table
{
    int size;
    hash_node **list;

} hash_table;

hash_table *create_hash_table(int size)
{
    hash_table *t = (hash_table *)malloc(sizeof(hash_table));
    t->size = size;
    t->list = (hash_node **)malloc(sizeof(hash_node *) * size);
    for (int i = 0; i < size; i++)
    {
        t->list[i] = NULL;
    }
    return t;
}

int hash_code(hash_table *t, int key)
{
    if (key < 0)
        return -(key % t->size);
    return key % t->size;
}

void hash_table_insert(hash_table *t, int key, uint64_t last_exe_cycle)
{
    int pos = hash_code(t, key);
    hash_node *list = t->list[pos];
    hash_node *newNode = (hash_node *)malloc(sizeof(hash_node));
    hash_node *temp = list;

    while (temp)
    {
        if (temp->key == key)
        {
            temp->stall_cycles += last_exe_cycle - temp->last_exe_cycle;
            temp->last_exe_cycle = last_exe_cycle;
            
            return;
        }
        temp = temp->next;
    }
    newNode->key = key;
    newNode->stall_cycles = 0;
    newNode->last_exe_cycle = last_exe_cycle;
    newNode->next = list;
    t->list[pos] = newNode;
}

hash_node *hash_table_lookup(hash_table *t, int key)
{
    int pos = hash_code(t, key);
    hash_node *list = t->list[pos];
    hash_node *temp = list;
    while (temp)
    {
        if (temp->key == key)
        {
            return temp;
        }
        temp = temp->next;
    }
    return NULL;
}

void print_hash_node(hash_table *t, int key)
{
    int pos = hash_code(t, key);
    hash_node *list = t->list[pos];
    hash_node *temp = list;
    while (temp)
    {
        if (temp->key == key)
        {
            printf("key: %d, stall_cycles: %ld\n", key, temp->stall_cycles);
        }
        temp = temp->next;
    }
}

#endif
