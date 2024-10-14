from astropy.coordinates import SkyCoord, ICRS, BarycentricTrueEcliptic
from astropy.coordinates import Angle
import astropy.units as u

# 定义恒星的赤经赤纬（时分秒和度分秒）
ra_hms = '13h18m23.15s'  # 赤经（时分秒）
dec_dms = '-18d18m56.79s'  # 赤纬（度分秒）

# 创建Angle对象
ra = Angle(ra_hms)
dec = Angle(dec_dms)

# 创建SkyCoord对象
star_coord = SkyCoord(ra=ra, dec=dec, frame='icrs')

# 将ICRS坐标转换为黄道坐标
ecliptic_coord = star_coord.transform_to(BarycentricTrueEcliptic)

# 获取黄经和黄纬
ecliptic_lon = ecliptic_coord.lon  # 黄经
ecliptic_lat = ecliptic_coord.lat  # 黄纬

print(f"黄经: {ecliptic_lon}")
print(f"黄纬: {ecliptic_lat}")
