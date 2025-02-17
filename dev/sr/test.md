(tkdev11) [less@devgpu115.cco2 ~/local/applied-ai/dev/sr (sr_kernel)]$ python usage.py
Launching kernel with blocks=1, threads_per_block=256, num_elements=12
Input tensor: tensor([ 0.3282, -0.4513, -1.0612,  0.1446, -0.8440, -1.4669, -0.7135, -0.6183,
        -2.2411,  2.1464,  1.4772, -1.3564], device='cuda:0')
Output tensor: tensor([ 0.3281, -0.4512, -1.0625,  0.1445, -0.8438, -1.4688, -0.7109, -0.6172,
        -2.2344,  2.1406,  1.4766, -1.3516], device='cuda:0',
       dtype=torch.bfloat16)
Output tensor dtype: torch.bfloat16
Success!
