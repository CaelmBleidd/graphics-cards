kernel void prefixSum(global const float *array, global float *result, uint n){

    const uint local_id = get_local_id(0);
    const uint parts = n / PART_SIZE;

    local float buf1[PART_SIZE];
    local float buf2[PART_SIZE];

    float acc = 0.0f;

    int last = 0;

    uint log = (uint) log2((float) PART_SIZE);
    for (uint part = 0; part < parts; ++part){
        buf1[local_id] = array[part * PART_SIZE + local_id];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0, max_id = 1; i < log; ++i, max_id <<= 1) {
            if ((i & 1)) {
                buf1[local_id] = local_id < max_id ? buf2[local_id] : buf2[local_id] + buf2[local_id - max_id];
                last &= 0;
            } else {
                buf2[local_id] = local_id < max_id ? buf1[local_id] : buf1[local_id] + buf1[local_id - max_id];
                last |= 1;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        result[part * PART_SIZE + local_id] = acc;
        if (last) {
             result[part * PART_SIZE + local_id] += buf2[local_id];
             acc += buf2[PART_SIZE - 1];
        } else {
             result[part * PART_SIZE + local_id] += buf1[local_id];
             acc += buf1[PART_SIZE - 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}