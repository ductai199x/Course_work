/* Reference implementation. */
extern "C" void compute_gold(int *, int *, int, int);

void compute_gold (int *input_data, int *histogram, int num_elements, int histogram_size)
{
    int i;
    for (i = 0; i < num_elements; i++)
        histogram[input_data[i]]++;
}

