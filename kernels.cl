/*
 * Based on:
 *  "HIGH-QUALITY LINEAR INTERPOLATION FOR DEMOSAICING OF BAYER-PATTERNED COLOR IMAGES"
 *  Henrique S. Malvar, Li-wei He, and Ross Cutler, 2004
 * And http://www.ipol.im/pub/art/2011/g_mhcd/
 *
 * Copyright 2015 Jason Newton <nevion@gmail.com>
 */

#include "clcommons/common.h"
#include "clcommons/image.h"

#ifndef OUTCHANNELS
#define OUTCHANNELS 3
#endif

#ifndef PIXELT
#define PIXELT uchar
#endif
#ifndef RGBPIXELT
#define RGBPIXELT PASTE(PIXELT, OUTCHANNELS)
#endif
#ifndef LDSPIXELT
#define LDSPIXELT int
#endif

typedef PIXELT PixelT;
typedef RGBPIXELT RGBPixelT;
typedef LDSPIXELT LDSPixelT;// for LDS's, having this large enough to prevent bank conflicts make's a large difference
#define kernel_size 5

#define tile_rows TILE_ROWS
#define tile_cols TILE_COLS
#define apron_rows (tile_rows + kernel_size - 1)
#define apron_cols (tile_cols + kernel_size - 1)

#define half_ksize  (kernel_size/2)
#define shalf_ksize ((int) half_ksize)
#define half_ksize_rem (kernel_size - half_ksize)
#define n_apron_fill_tasks (apron_rows * apron_cols)
#define n_tile_pixels  (tile_rows * tile_cols)

#define pixel_at(type, basename, r, c) image_pixel_at(type, PASTE2(basename, _p), im_rows, im_cols, PASTE2(basename, _pitch), (r), (c))
#define tex2D_at(type, basename, r, c) image_tex2D(type, PASTE2(basename, _p), im_rows, im_cols, PASTE2(basename, _pitch), (r), (c), ADDRESS_REFLECT_BORDER_EXCLUSIVE)
#define apron_pixel(_t_r, _t_c) apron[(_t_r)+ half_ksize][(_t_c) + half_ksize]

enum pattern_t{
    RGGB = 0,
    GRBG = 1,
    GBRG = 2,
    BGGR = 3
};

//this version takes a tile (z=1) and each tile job does 4 line median sorts
__kernel __attribute__((reqd_work_group_size(TILE_COLS, TILE_ROWS, 1)))
void malvar_he_cutler_demosaic(uint im_rows, uint im_cols,
    __global const uchar *input_image_p /* PixelT */, uint input_image_pitch, __global uchar * output_image_p /*RGBPixelT*/, uint output_image_pitch, int bayer_pattern){
    const uint tile_col_blocksize = get_local_size(0);
    const uint tile_row_blocksize = get_local_size(1);
    const uint tile_col_block = get_group_id(0) + get_global_offset(0) / tile_col_blocksize;
    const uint tile_row_block = get_group_id(1) + get_global_offset(1) / tile_row_blocksize;
    const uint tile_col = get_local_id(0);
    const uint tile_row = get_local_id(1);
    const uint g_c = get_global_id(0);
    const uint g_r = get_global_id(1);
    const bool valid_pixel_task = (g_r < im_rows) & (g_c < im_cols);

    __local LDSPixelT apron[apron_rows][apron_cols];

    const uint tile_flat_id = tile_row * tile_cols + tile_col;
    for(uint apron_fill_task_id = tile_flat_id; apron_fill_task_id < n_apron_fill_tasks; apron_fill_task_id += n_tile_pixels){
        const uint apron_read_row = apron_fill_task_id / apron_cols;
        const uint apron_read_col = apron_fill_task_id % apron_cols;
        const int ag_c = ((int)(apron_read_col + tile_col_block * tile_col_blocksize)) - shalf_ksize;
        const int ag_r = ((int)(apron_read_row + tile_row_block * tile_row_blocksize)) - shalf_ksize;

        apron[apron_read_row][apron_read_col] = tex2D_at(PixelT, input_image, ag_r, ag_c);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //valid tasks read from [half_ksize, (tile_rows|tile_cols) + kernel_size - 1)
    const uint a_c = tile_col + half_ksize;
    const uint a_r = tile_row + half_ksize;
    assert_val(a_c >= 0 && a_c < apron_cols, a_c);
    assert_val(a_r >= 0 && a_r < apron_rows, a_r);
    //const PixelT pixel = apron[a_r][a_c];

    //note the following formulas are col, row convention and uses i,j - this is done to preserve readability with the originating paper
    #define i a_c
    #define j a_r
    #define F(i, j) apron_pixel(j, i)

    const int G_at_blue_or_red = (4*F(i, j) + 2*(F(i-1,j) + F(i+1,j) + F(i,j-1) + F(i,j+1)) - 1*(F(i-2,j) + F(i+2,j) + F(i,j-2) + F(i,j+2))) / 8;

    const int R_or_B_at_green_in_red_row = (
                10*F(i,j)
               + F(i,j-2) + F(i,j+2)
               - 2*((F(i-1,j-1) + F(i+1,j-1) + F(i-1,j+1) + F(i+1,j+1)) + F(i-2,j) + F(i+2,j))
               + 8*(F(i-1,j) + F(i+1,j))
               ) / 16;

    const int R_or_B_at_green_in_blue_row = (
                 10*F(i,j)
               + F(i-2,j) + F(i+2,j)
               - 2*((F(i-1,j-1) + F(i+1,j-1) + F(i-1,j+1) + F(i+1,j+1)) + F(i,j-2) + F(i,j+2))
               + 8*(F(i,j-1) + F(i,j+1))
               ) / 16;

    const int R_at_B_or_B_at_R = (
                 12*F(i,j)
                - 3*(F(i-2,j) + F(i+2,j) + F(i,j-2) + F(i,j+2))
                + 4*(F(i-1,j-1) + F(i+1,j-1) + F(i-1,j+1) + F(i+1,j+1))) / 16;

    const int Fij = F(i,j);
    #undef F
    #undef j
    #undef i
    //RGGB -> RedXY = (0, 0), GreenXY1 = (1, 0), GreenXY2 = (0, 1), BlueXY = (1, 1)
    //GRBG -> RedXY = (1, 0), GreenXY1 = (0, 0), GreenXY2 = (1, 1), BlueXY = (0, 1)
    //GBRG -> RedXY = (0, 1), GreenXY1 = (0, 0), GreenXY2 = (1, 1), BlueXY = (1, 0)
    //BGGR -> RedXY = (1, 1), GreenXY1 = (1, 0), GreenXY2 = (0, 1), BlueXY = (0, 0)
    const int r_mod_2 = g_r & 1;
    const int c_mod_2 = g_c & 1;
    #define is_rggb (bayer_pattern == RGGB)
    #define is_grbg (bayer_pattern == GRBG)
    #define is_gbrg (bayer_pattern == GBRG)
    #define is_bggr (bayer_pattern == BGGR)


    const int red_col = is_rggb | is_bggr;
    const int red_row = is_gbrg | is_bggr;
    const int blue_col = is_rggb | is_gbrg;
    const int blue_row = is_rggb | is_grbg;

    const int in_red_row = r_mod_2 == red_row;
    const int in_blue_row = r_mod_2 == blue_row;
    const int is_red_pixel = (r_mod_2 == red_row) & (c_mod_2 == red_col);
    const int is_blue_pixel = (r_mod_2 == blue_row) & (c_mod_2 == blue_col);
    const int is_green_pixel = !(is_red_pixel | is_blue_pixel);

    const uchar R = convert_uchar_sat(
        is_red_pixel * Fij +
        R_or_B_at_green_in_red_row * (is_green_pixel & in_red_row) +
        R_or_B_at_green_in_blue_row * (is_green_pixel & in_blue_row) +
        R_at_B_or_B_at_R * is_blue_pixel
    );
    const uchar B = convert_uchar_sat(
        is_blue_pixel * Fij +
        R_or_B_at_green_in_red_row * (is_green_pixel & in_red_row) +
        R_or_B_at_green_in_blue_row * (is_green_pixel & in_blue_row) +
        R_at_B_or_B_at_R * is_red_pixel
    );
    const uchar G = convert_uchar_sat(is_green_pixel * Fij + G_at_blue_or_red * (!is_green_pixel));

    if(valid_pixel_task){
#if OUTCHANNELS == 3
        const RGBPixelT output = (RGBPIXELT)(R, G, B);
#elif OUTCHANNELS == 4
        const RGBPixelT output = (RGBPixelT)(R, G, B, 0);
#else
#error "Unsupported number of output channels"
#endif
        pixel_at(RGBPixelT, output_image, g_r, g_c) = output;
    }
}
