__kernel void anisodiff(__global const uchar * srcptr, int srcstep, int srcoffset,
                        __global uchar * dstptr, int dststep, int dstoffset,
                        int rows, int cols, __constant float* exptab, float alpha)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if( x < cols && y < rows )
    {
        int yofs = y*dststep + x*3;
        int xofs = y*srcstep + x*3;
        float4 s = 0.f;

        float4 c = (float4)(srcptr[xofs], srcptr[xofs+1], srcptr[xofs+2], 0.f);
        float4 delta, adelta;
        float w;

        #define UPDATE_SUM(xofs1) \
            delta = (float4)(srcptr[xofs + xofs1], srcptr[xofs + xofs1 + 1], srcptr[xofs + xofs1 + 2], 0.f) - c; \
            adelta = fabs(delta); \
            w = exptab[convert_int(adelta.x + adelta.y + adelta.z)]; \
            s += delta*w

        UPDATE_SUM(3);
        UPDATE_SUM(-3);
        UPDATE_SUM(-srcstep-3);
        UPDATE_SUM(-srcstep);
        UPDATE_SUM(-srcstep+3);
        UPDATE_SUM(srcstep-3);
        UPDATE_SUM(srcstep);
        UPDATE_SUM(srcstep+3);

        s = s*alpha + c;
        uchar4 d = convert_uchar4_sat(convert_int4_rte(s));
        dstptr[yofs] = d.x;
        dstptr[yofs+1] = d.y;
        dstptr[yofs+2] = d.z;
    }
}
