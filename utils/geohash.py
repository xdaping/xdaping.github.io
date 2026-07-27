#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:52

class Geohash(object):
    _base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    _base32_map = {}
    for i in range(len(_base32)):
        _base32_map[_base32[i]] = i
    del i

    def _float_hex_to_int(self, f):
        if f < -1.0 or f >= 1.0:
            return None

        if f == 0.0:
            return 1, 1

        h = f.hex()
        x = h.find("0x1.")
        assert (x >= 0)
        p = h.find("p")
        assert (p > 0)

        half_len = len(h[x + 4:p]) * 4 - int(h[p + 1:])
        if x == 0:
            r = (1 << half_len) + ((1 << (len(h[x + 4:p]) * 4)) + int(h[x + 4:p], 16))
        else:
            r = (1 << half_len) - ((1 << (len(h[x + 4:p]) * 4)) + int(h[x + 4:p], 16))

        return r, half_len + 1

    def _int_to_float_hex(self, i, l):
        if l == 0:
            return -1.0

        half = 1 << (l - 1)
        s = int((l + 3) / 4)
        if i >= half:
            i = i - half
            return float.fromhex(("0x0.%0" + str(s) + "xp1") % (i << (s * 4 - l),))
        else:
            i = half - i
            return float.fromhex(("-0x0.%0" + str(s) + "xp1") % (i << (s * 4 - l),))

    def _encode_i2c(self, lat, lon, lat_length, lon_length):
        precision = int((lat_length + lon_length) / 5)
        if lat_length < lon_length:
            a = lon
            b = lat
        else:
            a = lat
            b = lon

        boost = (0, 1, 4, 5, 16, 17, 20, 21)
        ret = ''
        for i in range(precision):
            ret += self._base32[(boost[a & 7] + (boost[b & 3] << 1)) & 0x1F]
            t = a >> 3
            a = b >> 2
            b = t

        return ret[::-1]

    def encode(self, latitude, longitude, precision=12):
        if latitude >= 90.0 or latitude < -90.0:
            raise Exception("invalid latitude.")
        while longitude < -180.0:
            longitude += 360.0
        while longitude >= 180.0:
            longitude -= 360.0

        # if _geohash:
        #     basecode = _geohash.encode(latitude, longitude)
        #     if len(basecode) > precision:
        #         return basecode[0:precision]
        #     return basecode + '0' * (precision - len(basecode))

        xprecision = precision + 1
        lat_length = lon_length = int(xprecision * 5 / 2)
        if xprecision % 2 == 1:
            lon_length += 1

        if hasattr(float, "fromhex"):
            a = self._float_hex_to_int(latitude / 90.0)
            o = self._float_hex_to_int(longitude / 180.0)
            if a[1] > lat_length:
                ai = a[0] >> (a[1] - lat_length)
            else:
                ai = a[0] << (lat_length - a[1])

            if o[1] > lon_length:
                oi = o[0] >> (o[1] - lon_length)
            else:
                oi = o[0] << (lon_length - o[1])

            return self._encode_i2c(ai, oi, lat_length, lon_length)[:precision]

        lat = latitude / 180.0
        lon = longitude / 360.0

        if lat > 0:
            lat = int((1 << lat_length) * lat) + (1 << (lat_length - 1))
        else:
            lat = (1 << lat_length - 1) - int((1 << lat_length) * (-lat))

        if lon > 0:
            lon = int((1 << lon_length) * lon) + (1 << (lon_length - 1))
        else:
            lon = (1 << lon_length - 1) - int((1 << lon_length) * (-lon))

        return self._encode_i2c(lat, lon, lat_length, lon_length)[:precision]

    def _decode_c2i(self, hashcode):
        lon = 0
        lat = 0
        bit_length = 0
        lat_length = 0
        lon_length = 0
        for i in hashcode:
            t = self._base32_map[i]
            if bit_length % 2 == 0:
                lon = lon << 3
                lat = lat << 2
                lon += (t >> 2) & 4
                lat += (t >> 2) & 2
                lon += (t >> 1) & 2
                lat += (t >> 1) & 1
                lon += t & 1
                lon_length += 3
                lat_length += 2
            else:
                lon = lon << 2
                lat = lat << 3
                lat += (t >> 2) & 4
                lon += (t >> 2) & 2
                lat += (t >> 1) & 2
                lon += (t >> 1) & 1
                lat += t & 1
                lon_length += 2
                lat_length += 3

            bit_length += 5

        return (lat, lon, lat_length, lon_length)

    def decode(self, hashcode, delta=False):
        '''
        decode a hashcode and get center coordinate, and distance between center and outer border
        '''
        (lat, lon, lat_length, lon_length) = self._decode_c2i(hashcode)

        if hasattr(float, "fromhex"):
            latitude_delta = 90.0 / (1 << lat_length)
            longitude_delta = 180.0 / (1 << lon_length)
            latitude = self._int_to_float_hex(lat, lat_length) * 90.0 + latitude_delta
            longitude = self._int_to_float_hex(lon, lon_length) * 180.0 + longitude_delta
            if delta:
                return latitude, longitude, latitude_delta, longitude_delta
            return latitude, longitude

        lat = (lat << 1) + 1
        lon = (lon << 1) + 1
        lat_length += 1
        lon_length += 1

        latitude = 180.0 * (lat - (1 << (lat_length - 1))) / (1 << lat_length)
        longitude = 360.0 * (lon - (1 << (lon_length - 1))) / (1 << lon_length)
        if delta:
            latitude_delta = 180.0 / (1 << lat_length)
            longitude_delta = 360.0 / (1 << lon_length)
            return latitude, longitude, latitude_delta, longitude_delta

        return latitude, longitude

    def decode_exactly(self, hashcode):
        return self.decode(hashcode, True)

    ## hashcode operations below

    def bbox(self, hashcode):
        '''
        decode a hashcode and get north, south, east and west border.
        '''
        (lat, lon, lat_length, lon_length) = self._decode_c2i(hashcode)
        ret = {}
        if lat_length:
            ret['n'] = 180.0 * (lat + 1 - (1 << (lat_length - 1))) / (1 << lat_length)
            ret['s'] = 180.0 * (lat - (1 << (lat_length - 1))) / (1 << lat_length)
        else:  # can't calculate the half with bit shifts (negative shift)
            ret['n'] = 90.0
            ret['s'] = -90.0

        if lon_length:
            ret['e'] = 360.0 * (lon + 1 - (1 << (lon_length - 1))) / (1 << lon_length)
            ret['w'] = 360.0 * (lon - (1 << (lon_length - 1))) / (1 << lon_length)
        else:  # can't calculate the half with bit shifts (negative shift)
            ret['e'] = 180.0
            ret['w'] = -180.0

        min_lat_for_block = ret['s']
        max_lat_for_block = ret['n']
        min_lng_for_block = ret['w']
        max_lng_for_block = ret['e']

        return (min_lng_for_block, min_lat_for_block, max_lng_for_block, max_lat_for_block)

    def neighbors(self, hashcode):
        # if _geohash and len(hashcode) < 25:
        #     return _geohash.neighbors(hashcode)
        (lat, lon, lat_length, lon_length) = self._decode_c2i(hashcode)
        ret = []
        tlat = lat
        for tlon in (lon - 1, lon + 1):
            ret.append(self._encode_i2c(tlat, tlon, lat_length, lon_length))

        tlat = lat + 1
        if not tlat >> lat_length:
            for tlon in (lon - 1, lon, lon + 1):
                ret.append(self._encode_i2c(tlat, tlon, lat_length, lon_length))

        tlat = lat - 1
        if tlat >= 0:
            for tlon in (lon - 1, lon, lon + 1):
                ret.append(self._encode_i2c(tlat, tlon, lat_length, lon_length))

        return ret


if __name__ == '__main__':
    res = Geohash().bbox('wtw1vmy7')
    print(res)