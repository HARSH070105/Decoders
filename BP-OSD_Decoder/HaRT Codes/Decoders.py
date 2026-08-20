from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bp_decoder import BpDecoder

def setup_all_decoders(HX, HZ, p, noise_model='depolarizing', osd_order=0):
    
    if noise_model == 'depolarizing':
        px = 2*p/3
        pz = 2*p/3
    elif noise_model == 'pure_x':
        px = p
        pz = 0.0  # Prevents LLR log(0) errors
    elif noise_model == 'pure_z':
        px = 0.0  
        pz = p
    elif noise_model == 'pure_y':
            px = p/2  
            pz = p/2
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    bp_x = BpDecoder(
        HZ,
        error_rate=px,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625, 
        max_iter=32,
        schedule='serial'
    )
    bp_z = BpDecoder(
        HX,
        error_rate=pz,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        max_iter=32,
        schedule='serial'
    )

    # ---------------------------------------------------------
    # BP-OSD Decoders
    # ---------------------------------------------------------
    osd_method_str = 'OSD_0' if osd_order == 0 else 'OSD_CS'
    
    osd_x = BpOsdDecoder(
        HZ,
        error_rate=px,
        bp_method='minimum_sum',      
        ms_scaling_factor=0.625, 
        max_iter=32,
        schedule='serial',
        osd_method=osd_method_str,
        osd_order=osd_order
    )
    osd_z = BpOsdDecoder(
        HX,
        error_rate=pz,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        max_iter=32,
        schedule='serial',
        osd_method=osd_method_str,
        osd_order=osd_order
    )
    
    return bp_x, bp_z, osd_x, osd_z