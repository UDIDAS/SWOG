#!/usr/bin/env python3
"""Parse the FLARE23 Metadata.zip central directory (central_tail.bin) for the IMAGE entries
(Metadata/images/FLARE23_*_0000.nii.gz) -> image offsets + compressed sizes, so we can byte-range
extract them from the remote zip without downloading the whole 87 GB. Handles zip64 offsets.

out: /scratch/ud3d4/acm_data/flare23_labels/image_index.json  {images:[[name,offset,csize], ...]}
"""
import json
import struct

LAB = "/scratch/ud3d4/acm_data/flare23_labels"
buf = open(f"{LAB}/central_tail.bin", "rb").read()


def zip64_fields(extra, unc_ff, csize_ff, off_ff):
    """Zip64 extra (0x0001) holds ONLY the fields whose CDH value was 0xFFFFFFFF, in this fixed order:
    uncompressed(8), compressed(8), local-header-offset(8), disk(4). Return (offset, csize) overrides."""
    i = 0
    off = csize = None
    while i + 4 <= len(extra):
        hid, sz = struct.unpack("<HH", extra[i:i + 4])
        if hid == 0x0001:
            body = extra[i + 4:i + 4 + sz]
            p = 0
            if unc_ff:
                p += 8                                   # uncompressed present -> skip
            if csize_ff and p + 8 <= len(body):
                csize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
            if off_ff and p + 8 <= len(body):
                off = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
            break
        i += 4 + sz
    return off, csize


images = []
i = 0
while True:
    j = buf.find(b"PK\x01\x02", i)
    if j < 0:
        break
    unc = struct.unpack("<I", buf[j + 24:j + 28])[0]
    csize = struct.unpack("<I", buf[j + 20:j + 24])[0]
    nlen = struct.unpack("<H", buf[j + 28:j + 30])[0]
    elen = struct.unpack("<H", buf[j + 30:j + 32])[0]
    clen = struct.unpack("<H", buf[j + 32:j + 34])[0]
    off = struct.unpack("<I", buf[j + 42:j + 46])[0]
    name = buf[j + 46:j + 46 + nlen].decode("latin1", "replace")
    extra = buf[j + 46 + nlen:j + 46 + nlen + elen]
    i = j + 46 + nlen + elen + clen
    if not name.endswith("_0000.nii.gz"):
        continue
    z_off, z_csize = zip64_fields(extra, unc == 0xFFFFFFFF, csize == 0xFFFFFFFF, off == 0xFFFFFFFF)
    if off == 0xFFFFFFFF and z_off is not None:
        off = z_off
    if csize == 0xFFFFFFFF and z_csize is not None:
        csize = z_csize
    images.append([name, int(off), int(csize)])

images.sort(key=lambda x: x[1])
json.dump({"images": images, "n": len(images)}, open(f"{LAB}/image_index.json", "w"))
tot = sum(c for _, _, c in images) / 1e9
print(f"{len(images)} image entries; total compressed ~{tot:.1f} GB")
print("offset range:", images[0][1], "..", images[-1][1])
print("sample:", images[0][0], "csize", images[0][2] / 1e6, "MB")
