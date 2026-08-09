def interpolate_colors(color1, color2, t=0.5):
    t = max(0.0, min(1.0, t)) 
    
    r = int(color1[0] * (1 - t) + color2[0] * t)
    g = int(color1[1] * (1 - t) + color2[1] * t)
    b = int(color1[2] * (1 - t) + color2[2] * t)
    
    return (r, g, b)