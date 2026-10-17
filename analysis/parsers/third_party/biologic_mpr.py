"""
BioLogic MPR Binary File Decoder.

This module decodes BioLogic EC-Lab modular binary (.mpr) files into raw numpy arrays.
Directly adapted from the open-source galvani library by Christopher Kerr and cellpy by cellpy developers
(see third_party_licenses/cellpy_LICENSE.txt).

-----------------------------------------------------------------------------
MIT License

Copyright (c) 2013-2024 Christopher Kerr
Copyright (c) 2016-2026 cellpy developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-----------------------------------------------------------------------------
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import numpy as np

mpr_label = b"BIO-LOGIC MODULAR FILE\x1a                         \x00\x00\x00\x00"

hdr_dtype = np.dtype(
    [
        ("shortname", "S10"),
        ("longname", "S25"),
        ("length", "<u4"),
        ("version", "<u4"),
        ("date", "S8"),
    ]
)

# Full BioLogic column definitions (509 column types from galvani / cellpy)
bl_dtypes = {1: ['u1', 'flags', 'int', 'mode', ''],
 2: ['u1', 'flags', 'int', 'ox/red', ''],
 3: ['u1', 'flags', 'int', 'error', ''],
 4: ['<f8', 'time', 's', 'time/s', 'test_time_txt'],
 5: ['<f4', 'control', 'V/mA', 'control/V/mA', ''],
 6: ['<f4', 'Ewe', 'V', 'Ewe/V', 'voltage_txt'],
 7: ['<f8', 'dQ', 'mAh', 'dq/mA.h', ''],
 8: ['<f4', 'I', 'mA', 'I/mA', 'current_txt'],
 9: ['<f4', 'Ece', '', 'Ece/V', 'ref_voltage_txt'],
 10: ['<f4', 'NN_10', '', '', ''],
 11: ['<f8', 'I', 'mA', '<I>/mA', ''],
 12: ['<f4', 'NN_12', '', '', ''],
 13: ['<f8', 'QminusQnull', 'mAh', '(Q-Qo)/mA.h', ''],
 14: ['<f4', 'NN_14', '', '', ''],
 15: ['<f4', 'NN_15', '', '', ''],
 16: ['<f4', 'Analog_IN_1', 'V', 'Analog IN 1/V', ''],
 17: ['<f4', 'Analog_IN_2', 'V', 'Analog IN 2/V', ''],
 18: ['<f4', 'NN_18', '', '', ''],
 19: ['<f4', 'control', 'V/mA', 'control/V', ''],
 20: ['<f4', 'control', 'mA', 'control/mA', ''],
 21: ['u1', 'flags', 'int', 'control changes', ''],
 22: ['<f4', 'NN_22', '', '', ''],
 23: ['<f8', 'dQ', 'mAh', 'dQ/mA.h', ''],
 24: ['<f8', 'cycleno', 'int', 'cycle number', 'cycle_index_txt'],
 25: ['<f4', 'NN_25', '', '', ''],
 26: ['<f4', 'Rapp', 'Ohm', 'Rapp/Ohm', 'internal_resistance_txt'],
 27: ['<f4', 'Ewe_Ece', 'V', 'Ewe-Ece/V', 'voltage_txt'],
 28: ['<f4', 'NN_28', '', '', ''],
 29: ['<f4', 'NN_29', '', '', ''],
 30: ['<f4', 'NN_30', '', '', ''],
 31: ['u1', 'flags', 'int', 'Ns changes', ''],
 32: ['<f4', 'freq', 'Hz', 'freq/Hz', 'frequency_txt'],
 33: ['<f4', 'absEwe', 'V', '|Ewe|/V', ''],
 34: ['<f4', 'absI', 'A', '|I|/A', ''],
 35: ['<f4', 'phaseZ', 'deg', 'Phase(Z)/deg', 'aci_phase_angle_txt'],
 36: ['<f4', 'absZ', 'Ohm', '|Z|/Ohm', 'amplitude_txt'],
 37: ['<f4', 'ReZ', 'Ohm', 'Re(Z)/Ohm', ''],
 38: ['<f4', 'negImZ', 'Ohm', '-Im(Z)/Ohm', ''],
 39: ['<u2', 'Irange', 'int', 'I Range', ''],
 40: ['<f4', 'NN_40', '', '', ''],
 41: ['<f4', 'NN_41', '', '', ''],
 42: ['<f4', 'NN_42', '', '', ''],
 43: ['<f4', 'NN_43', '', '', ''],
 44: ['<f4', 'NN_44', '', '', ''],
 45: ['<f4', 'NN_45', '', '', ''],
 46: ['<f4', 'NN_46', '', '', ''],
 47: ['<f4', 'NN_47', '', '', ''],
 48: ['<f4', 'NN_48', '', '', ''],
 49: ['<f4', 'NN_49', '', '', ''],
 50: ['<f4', 'NN_50', '', '', ''],
 51: ['<f4', 'NN_51', '', '', ''],
 52: ['<f4', 'NN_52', '', '', ''],
 53: ['<f4', 'NN_53', '', '', ''],
 54: ['<f4', 'NN_54', '', '', ''],
 55: ['<f4', 'NN_55', '', '', ''],
 56: ['<f4', 'NN_56', '', '', ''],
 57: ['<f4', 'NN_57', '', '', ''],
 58: ['<f4', 'NN_58', '', '', ''],
 59: ['<f4', 'NN_59', '', '', ''],
 60: ['<f4', 'NN_60', '', '', ''],
 61: ['<f4', 'NN_61', '', '', ''],
 62: ['<f4', 'NN_62', '', '', ''],
 63: ['<f4', 'NN_63', '', '', ''],
 64: ['<f4', 'NN_64', '', '', ''],
 65: ['u1', 'flags', 'int', 'counter inc.', ''],
 66: ['<f4', 'NN_66', '', '', ''],
 67: ['<f4', 'NN_67', '', '', ''],
 68: ['<f4', 'NN_68', '', '', ''],
 69: ['<f4', 'R', 'Ohm', 'R/Ohm', ''],
 70: ['<f4', 'P', 'W', 'P/W', ''],
 71: ['<f4', 'QminusQnull', 'C', '', ''],
 72: ['<f4', 'NN_72', '', '', ''],
 73: ['<f4', 'NN_73', '', '', ''],
 74: ['<f8', 'Energy', 'W.h', '|Energy|/W.h', 'charge_energy_txt'],
 75: ['<f4', 'Analog_OUT', 'V', 'Analog OUT/V', ''],
 76: ['<f4', 'I', 'mA', '<I>/mA', ''],
 77: ['<f4', 'Ewe', 'V', '<Ewe>/V', ''],
 78: ['<f4', 'Cs_2', 'F_substr_2', 'Cs-2/F-2', ''],
 79: ['<f4', 'NN_79', '', '', ''],
 80: ['<f4', 'NN_80', '', '', ''],
 81: ['<f4', 'NN_81', '', '', ''],
 82: ['<f4', 'NN_82', '', '', ''],
 83: ['<f4', 'NN_83', '', '', ''],
 84: ['<f4', 'NN_84', '', '', ''],
 85: ['<f4', 'NN_85', '', '', ''],
 86: ['<f4', 'NN_86', '', '', ''],
 87: ['<f4', 'NN_87', '', '', ''],
 88: ['<f4', 'NN_88', '', '', ''],
 89: ['<f4', 'NN_89', '', '', ''],
 90: ['<f4', 'NN_90', '', '', ''],
 91: ['<f4', 'NN_91', '', '', ''],
 92: ['<f4', 'NN_92', '', '', ''],
 93: ['<f4', 'NN_93', '', '', ''],
 94: ['<f4', 'NN_94', '', '', ''],
 95: ['<f4', 'NN_95', '', '', ''],
 96: ['<f4', 'absEce', 'V', '|Ece|/V', ''],
 97: ['<f4', 'NN_97', '', '', ''],
 98: ['<f4', 'phaseZce', 'deg', 'Phase(Zce)/deg', 'ref_aci_phase_angle_txt'],
 99: ['<f4', 'absZce', 'Ohm', '|Zce|/Ohm', ''],
 100: ['<f4', 'ReZce', 'Ohm', 'Re(Zce)/Ohm', ''],
 101: ['<f4', 'neg_ImZce', 'Ohm', '-Im(Zce)/Ohm', ''],
 102: ['<f4', 'NN_102', '', '', ''],
 103: ['<f4', 'NN_103', '', '', ''],
 104: ['<f4', 'NN_104', '', '', ''],
 105: ['<f4', 'NN_105', '', '', ''],
 106: ['<f4', 'NN_106', '', '', ''],
 107: ['<f4', 'NN_107', '', '', ''],
 108: ['<f4', 'NN_108', '', '', ''],
 109: ['<f4', 'NN_109', '', '', ''],
 110: ['<f4', 'NN_110', '', '', ''],
 111: ['<f4', 'NN_111', '', '', ''],
 112: ['<f4', 'NN_112', '', '', ''],
 113: ['<f4', 'NN_113', '', '', ''],
 114: ['<f4', 'NN_114', '', '', ''],
 115: ['<f4', 'NN_115', '', '', ''],
 116: ['<f4', 'NN_116', '', '', ''],
 117: ['<f4', 'NN_117', '', '', ''],
 118: ['<f4', 'NN_118', '', '', ''],
 119: ['<f4', 'NN_119', '', '', ''],
 120: ['<f4', 'NN_120', '', '', ''],
 121: ['<f4', 'NN_121', '', '', ''],
 122: ['<f4', 'NN_122', '', '', ''],
 123: ['<f8', 'Energy_charge', 'Wh', 'Energy charge/W.h', 'charge_energy_txt'],
 124: ['<f8', 'Energy_discharge', 'Wh', 'Energy discharge/W.h', 'discharge_energy_txt'],
 125: ['<f8', 'capacitance_charge', 'microF', 'Capacitance charge/F', ''],
 126: ['<f8', 'capacitance_discharge', 'F', 'Capacitance discharge/F', ''],
 127: ['<f4', 'NN_127', '', '', ''],
 128: ['<f4', 'NN_128', '', '', ''],
 129: ['<f4', 'NN_129', '', '', ''],
 130: ['<f4', 'NN_130', '', '', ''],
 131: ['<u2', 'Ns', '', 'Ns', 'step_index_txt'],
 132: ['<f4', 'NN_132', '', '', ''],
 133: ['<f4', 'NN_133', '', '', ''],
 134: ['<f4', 'NN_134', '', '', ''],
 135: ['<f4', 'NN_135', '', '', ''],
 136: ['<f4', 'NN_136', '', '', ''],
 137: ['<f4', 'NN_137', '', '', ''],
 138: ['<f4', 'NN_138', '', '', ''],
 139: ['<f4', 'NN_139', '', '', ''],
 140: ['<f4', 'NN_140', '', '', ''],
 141: ['<f4', 'NN_141', '', '', ''],
 142: ['<f4', 'NN_142', '', '', ''],
 143: ['<f4', 'NN_143', '', '', ''],
 144: ['<f4', 'NN_144', '', '', ''],
 145: ['<f4', 'NN_145', '', '', ''],
 146: ['<f4', 'NN_146', '', '', ''],
 147: ['<f4', 'NN_147', '', '', ''],
 148: ['<f4', 'NN_148', '', '', ''],
 149: ['<f4', 'NN_149', '', '', ''],
 150: ['<f4', 'NN_150', '', '', ''],
 151: ['<f4', 'NN_151', '', '', ''],
 152: ['<f4', 'NN_152', '', '', ''],
 153: ['<f4', 'NN_153', '', '', ''],
 154: ['<f4', 'NN_154', '', '', ''],
 155: ['<f4', 'NN_155', '', '', ''],
 156: ['<f4', 'NN_156', '', '', ''],
 157: ['<f4', 'NN_157', '', '', ''],
 158: ['<f4', 'NN_158', '', '', ''],
 159: ['<f4', 'NN_159', '', '', ''],
 160: ['<f4', 'NN_160', '', '', ''],
 161: ['<f4', 'NN_161', '', '', ''],
 162: ['<f4', 'NN_162', '', '', ''],
 163: ['<f4', 'abs_Estack', 'V', '|Estack|/V', ''],
 164: ['<f4', 'NN_164', '', '', ''],
 165: ['<f4', 'NN_165', '', '', ''],
 166: ['<f4', 'NN_166', '', '', ''],
 167: ['<f4', 'NN_167', '', '', ''],
 168: ['<f4', 'Rcmp', 'Ohm', 'Rcmp/Ohm', ''],
 169: ['<f4', 'Cs', 'F', 'Cs/F', ''],
 170: ['<f4', 'NN_170', '', '', ''],
 171: ['<f4', 'NN_171', '', '', ''],
 172: ['<f4', 'Cp', 'F', 'Cp/F', ''],
 173: ['<f4', 'Cp_2', 'F_substr_2', 'Cp-2/F-2', ''],
 174: ['<f4', 'avr_Ewe', 'V', '<Ewe>/V', ''],
 175: ['<f4', 'NN_175', '', '', ''],
 176: ['<f4', 'NN_176', '', '', ''],
 177: ['<f4', 'NN_177', '', '', ''],
 178: ['<f4', 'Q_Qo', 'C', '(Q-Qo)/C', ''],
 179: ['<f4', 'dQ', 'C', 'dQ/C', ''],
 180: ['<f4', 'NN_180', '', '', ''],
 181: ['<f4', 'NN_181', '', '', ''],
 182: ['<f4', 'NN_182', '', '', ''],
 183: ['<f4', 'NN_183', '', '', ''],
 184: ['<f4', 'NN_184', '', '', ''],
 185: ['<f4', 'NN_185', '', '', ''],
 186: ['<f4', 'NN_186', '', '', ''],
 187: ['<f4', 'NN_187', '', '', ''],
 188: ['<f4', 'NN_188', '', '', ''],
 189: ['<f4', 'NN_189', '', '', ''],
 190: ['<f4', 'NN_190', '', '', ''],
 191: ['<f4', 'NN_191', '', '', ''],
 192: ['<f4', 'NN_192', '', '', ''],
 193: ['<f4', 'NN_193', '', '', ''],
 194: ['<f4', 'NN_194', '', '', ''],
 195: ['<f4', 'NN_195', '', '', ''],
 196: ['<f4', 'NN_196', '', '', ''],
 197: ['<f4', 'NN_197', '', '', ''],
 198: ['<f4', 'NN_198', '', '', ''],
 199: ['<f4', 'NN_199', '', '', ''],
 200: ['<f4', 'NN_200', '', '', ''],
 201: ['<f4', 'NN_201', '', '', ''],
 202: ['<f4', 'NN_202', '', '', ''],
 203: ['<f4', 'NN_203', '', '', ''],
 204: ['<f4', 'NN_204', '', '', ''],
 205: ['<f4', 'NN_205', '', '', ''],
 206: ['<f4', 'NN_206', '', '', ''],
 207: ['<f4', 'NN_207', '', '', ''],
 208: ['<f4', 'NN_208', '', '', ''],
 209: ['<f4', 'NN_209', '', '', ''],
 210: ['<f4', 'NN_210', '', '', ''],
 211: ['<f8', 'Q_charge/discharge/mA.h', '', 'Q charge/discharge/mA.h', ''],
 212: ['<u4', 'half_cycle', '', 'half cycle', ''],
 213: ['<u4', 'z_cycle', '', 'z cycle', ''],
 214: ['<f4', 'NN_214', '', '', ''],
 215: ['<f4', 'NN_215', '', '', ''],
 216: ['<f4', 'NN_216', '', '', ''],
 217: ['<f4', 'THD_Ewe', 'percent', 'THD Ewe/%', ''],
 218: ['<f4', 'THD_I', 'percent', 'THD I/%', ''],
 219: ['<f4', 'NN_219', '', '', ''],
 220: ['<f4', 'NSD_Ewe', 'percent', 'NSD Ewe/%', ''],
 221: ['<f4', 'NSD_I', 'percent', 'NSD I/%', ''],
 222: ['<f4', 'NN_222', '', '', ''],
 223: ['<f4', 'NSR_Ewe', 'percent', 'NSR Ewe/%', ''],
 224: ['<f4', 'NSR_I', 'percent', 'NSR I/%', ''],
 225: ['<f4', 'NN_225', '', '', ''],
 226: ['<f4', 'NN_226', '', '', ''],
 227: ['<f4', 'NN_227', '', '', ''],
 228: ['<f4', 'NN_228', '', '', ''],
 229: ['<f4', 'NN_229', '', '', ''],
 230: ['<f4', 'abs_Ewe_h2', 'V', '|Ewe h2|/V', ''],
 231: ['<f4', 'abs_Ewe_h3', 'V', '|Ewe h3|/V', ''],
 232: ['<f4', 'abs_Ewe_h4', 'V', '|Ewe h4|/V', ''],
 233: ['<f4', 'abs_Ewe_h5', 'V', '|Ewe h5|/V', ''],
 234: ['<f4', 'abs_Ewe_h6', 'V', '|Ewe h6|/V', ''],
 235: ['<f4', 'abs_Ewe_h7', 'V', '|Ewe h7|/V', ''],
 236: ['<f4', 'abs_I_h2', 'A', '|I h2|/A', ''],
 237: ['<f4', 'abs_I_h3', 'A', '|I h3|/A', ''],
 238: ['<f4', 'abs_I_h4', 'A', '|I h4|/A', ''],
 239: ['<f4', 'abs_I_h5', 'A', '|I h5|/A', ''],
 240: ['<f4', 'abs_I_h6', 'A', '|I h6|/A', ''],
 241: ['<f4', 'abs_I_h7', 'A', '|I h7|/A', ''],
 242: ['<f4', 'abs_E2', 'V', '|E2|/V', ''],
 243: ['<f4', 'NN_243', '', '', ''],
 244: ['<f4', 'NN_244', '', '', ''],
 245: ['<f4', 'NN_245', '', '', ''],
 246: ['<f4', 'NN_246', '', '', ''],
 247: ['<f4', 'NN_247', '', '', ''],
 248: ['<f4', 'NN_248', '', '', ''],
 249: ['<f4', 'NN_249', '', '', ''],
 250: ['<f4', 'NN_250', '', '', ''],
 251: ['<f4', 'NN_251', '', '', ''],
 252: ['<f4', 'NN_252', '', '', ''],
 253: ['<f4', 'NN_253', '', '', ''],
 254: ['<f4', 'NN_254', '', '', ''],
 255: ['<f4', 'NN_255', '', '', ''],
 256: ['<f4', 'NN_256', '', '', ''],
 257: ['<f4', 'NN_257', '', '', ''],
 258: ['<f4', 'NN_258', '', '', ''],
 259: ['<f4', 'NN_259', '', '', ''],
 260: ['<f4', 'NN_260', '', '', ''],
 261: ['<f4', 'NN_261', '', '', ''],
 262: ['<f4', 'NN_262', '', '', ''],
 263: ['<f4', 'NN_263', '', '', ''],
 264: ['<f4', 'NN_264', '', '', ''],
 265: ['<f4', 'NN_265', '', '', ''],
 266: ['<f4', 'NN_266', '', '', ''],
 267: ['<f4', 'NN_267', '', '', ''],
 268: ['<f4', 'NN_268', '', '', ''],
 269: ['<f4', 'NN_269', '', '', ''],
 270: ['<f4', 'NN_270', '', '', ''],
 271: ['<f4', 'PhaseZ1', 'deg', 'Phase(Z1) / deg', ''],
 272: ['<f4', 'PhaseZ2', 'deg', 'Phase(Z2) / deg', ''],
 273: ['<f4', 'NN_273', '', '', ''],
 274: ['<f4', 'NN_274', '', '', ''],
 275: ['<f4', 'NN_275', '', '', ''],
 276: ['<f4', 'NN_276', '', '', ''],
 277: ['<f4', 'NN_277', '', '', ''],
 278: ['<f4', 'NN_278', '', '', ''],
 279: ['<f4', 'NN_279', '', '', ''],
 280: ['<f4', 'NN_280', '', '', ''],
 281: ['<f4', 'NN_281', '', '', ''],
 282: ['<f4', 'NN_282', '', '', ''],
 283: ['<f4', 'NN_283', '', '', ''],
 284: ['<f4', 'NN_284', '', '', ''],
 285: ['<f4', 'NN_285', '', '', ''],
 286: ['<f4', 'NN_286', '', '', ''],
 287: ['<f4', 'NN_287', '', '', ''],
 288: ['<f4', 'NN_288', '', '', ''],
 289: ['<f4', 'NN_289', '', '', ''],
 290: ['<f4', 'NN_290', '', '', ''],
 291: ['<f4', 'NN_291', '', '', ''],
 292: ['<f4', 'NN_292', '', '', ''],
 293: ['<f4', 'NN_293', '', '', ''],
 294: ['<f4', 'NN_294', '', '', ''],
 295: ['<f4', 'NN_295', '', '', ''],
 296: ['<f4', 'NN_296', '', '', ''],
 297: ['<f4', 'NN_297', '', '', ''],
 298: ['<f4', 'NN_298', '', '', ''],
 299: ['<f4', 'NN_299', '', '', ''],
 300: ['<f4', 'NN_300', '', '', ''],
 301: ['<f4', 'abs_Z1', 'Ohm', '|Z1|/Ohm', ''],
 302: ['<f4', 'abs_Z2', 'Ohm', '|Z2|/Ohm', ''],
 303: ['<f4', 'NN_303', '', '', ''],
 304: ['<f4', 'NN_304', '', '', ''],
 305: ['<f4', 'NN_305', '', '', ''],
 306: ['<f4', 'NN_306', '', '', ''],
 307: ['<f4', 'NN_307', '', '', ''],
 308: ['<f4', 'NN_308', '', '', ''],
 309: ['<f4', 'NN_309', '', '', ''],
 310: ['<f4', 'NN_310', '', '', ''],
 311: ['<f4', 'NN_311', '', '', ''],
 312: ['<f4', 'NN_312', '', '', ''],
 313: ['<f4', 'NN_313', '', '', ''],
 314: ['<f4', 'NN_314', '', '', ''],
 315: ['<f4', 'NN_315', '', '', ''],
 316: ['<f4', 'NN_316', '', '', ''],
 317: ['<f4', 'NN_317', '', '', ''],
 318: ['<f4', 'NN_318', '', '', ''],
 319: ['<f4', 'NN_319', '', '', ''],
 320: ['<f4', 'NN_320', '', '', ''],
 321: ['<f4', 'NN_321', '', '', ''],
 322: ['<f4', 'NN_322', '', '', ''],
 323: ['<f4', 'NN_323', '', '', ''],
 324: ['<f4', 'NN_324', '', '', ''],
 325: ['<f4', 'NN_325', '', '', ''],
 326: ['<f4', 'NN_326', '', '', ''],
 327: ['<f4', 'NN_327', '', '', ''],
 328: ['<f4', 'NN_328', '', '', ''],
 329: ['<f4', 'NN_329', '', '', ''],
 330: ['<f4', 'NN_330', '', '', ''],
 331: ['<f4', 'ReZ1', 'Ohm', 'Re(Z1)/Ohm', ''],
 332: ['<f4', 'ReZ2', 'Ohm', 'Re(Z2)/Ohm', ''],
 333: ['<f4', 'NN_333', '', '', ''],
 334: ['<f4', 'NN_334', '', '', ''],
 335: ['<f4', 'NN_335', '', '', ''],
 336: ['<f4', 'NN_336', '', '', ''],
 337: ['<f4', 'NN_337', '', '', ''],
 338: ['<f4', 'NN_338', '', '', ''],
 339: ['<f4', 'NN_339', '', '', ''],
 340: ['<f4', 'NN_340', '', '', ''],
 341: ['<f4', 'NN_341', '', '', ''],
 342: ['<f4', 'NN_342', '', '', ''],
 343: ['<f4', 'NN_343', '', '', ''],
 344: ['<f4', 'NN_344', '', '', ''],
 345: ['<f4', 'NN_345', '', '', ''],
 346: ['<f4', 'NN_346', '', '', ''],
 347: ['<f4', 'NN_347', '', '', ''],
 348: ['<f4', 'NN_348', '', '', ''],
 349: ['<f4', 'NN_349', '', '', ''],
 350: ['<f4', 'NN_350', '', '', ''],
 351: ['<f4', 'NN_351', '', '', ''],
 352: ['<f4', 'NN_352', '', '', ''],
 353: ['<f4', 'NN_353', '', '', ''],
 354: ['<f4', 'NN_354', '', '', ''],
 355: ['<f4', 'NN_355', '', '', ''],
 356: ['<f4', 'NN_356', '', '', ''],
 357: ['<f4', 'NN_357', '', '', ''],
 358: ['<f4', 'NN_358', '', '', ''],
 359: ['<f4', 'NN_359', '', '', ''],
 360: ['<f4', 'NN_360', '', '', ''],
 361: ['<f4', 'neg_ImZ1', 'Ohm', '-Im(Z1)/Ohm', ''],
 362: ['<f4', 'neg_ImZ2', 'Ohm', '-Im(Z2)/Ohm', ''],
 363: ['<f4', 'NN_363', '', '', ''],
 364: ['<f4', 'NN_364', '', '', ''],
 365: ['<f4', 'NN_365', '', '', ''],
 366: ['<f4', 'NN_366', '', '', ''],
 367: ['<f4', 'NN_367', '', '', ''],
 368: ['<f4', 'NN_368', '', '', ''],
 369: ['<f4', 'NN_369', '', '', ''],
 370: ['<f4', 'NN_370', '', '', ''],
 371: ['<f4', 'NN_371', '', '', ''],
 372: ['<f4', 'NN_372', '', '', ''],
 373: ['<f4', 'NN_373', '', '', ''],
 374: ['<f4', 'NN_374', '', '', ''],
 375: ['<f4', 'NN_375', '', '', ''],
 376: ['<f4', 'NN_376', '', '', ''],
 377: ['<f4', 'NN_377', '', '', ''],
 378: ['<f4', 'NN_378', '', '', ''],
 379: ['<f4', 'NN_379', '', '', ''],
 380: ['<f4', 'NN_380', '', '', ''],
 381: ['<f4', 'NN_381', '', '', ''],
 382: ['<f4', 'NN_382', '', '', ''],
 383: ['<f4', 'NN_383', '', '', ''],
 384: ['<f4', 'NN_384', '', '', ''],
 385: ['<f4', 'NN_385', '', '', ''],
 386: ['<f4', 'NN_386', '', '', ''],
 387: ['<f4', 'NN_387', '', '', ''],
 388: ['<f4', 'NN_388', '', '', ''],
 389: ['<f4', 'NN_389', '', '', ''],
 390: ['<f4', 'NN_390', '', '', ''],
 391: ['<f4', 'avr_E1', 'V', '<E1>/V', ''],
 392: ['<f4', 'avr_E2', 'V', '<E2>/V', ''],
 393: ['<f4', 'NN_393', '', '', ''],
 394: ['<f4', 'NN_394', '', '', ''],
 395: ['<f4', 'NN_395', '', '', ''],
 396: ['<f4', 'NN_396', '', '', ''],
 397: ['<f4', 'NN_397', '', '', ''],
 398: ['<f4', 'NN_398', '', '', ''],
 399: ['<f4', 'NN_399', '', '', ''],
 400: ['<f4', 'NN_400', '', '', ''],
 401: ['<f4', 'NN_401', '', '', ''],
 402: ['<f4', 'NN_402', '', '', ''],
 403: ['<f4', 'NN_403', '', '', ''],
 404: ['<f4', 'NN_404', '', '', ''],
 405: ['<f4', 'NN_405', '', '', ''],
 406: ['<f4', 'NN_406', '', '', ''],
 407: ['<f4', 'NN_407', '', '', ''],
 408: ['<f4', 'NN_408', '', '', ''],
 409: ['<f4', 'NN_409', '', '', ''],
 410: ['<f4', 'NN_410', '', '', ''],
 411: ['<f4', 'NN_411', '', '', ''],
 412: ['<f4', 'NN_412', '', '', ''],
 413: ['<f4', 'NN_413', '', '', ''],
 414: ['<f4', 'NN_414', '', '', ''],
 415: ['<f4', 'NN_415', '', '', ''],
 416: ['<f4', 'NN_416', '', '', ''],
 417: ['<f4', 'NN_417', '', '', ''],
 418: ['<f4', 'NN_418', '', '', ''],
 419: ['<f4', 'NN_419', '', '', ''],
 420: ['<f4', 'NN_420', '', '', ''],
 421: ['<f4', 'NN_421', '', '', ''],
 422: ['<f4', 'PhaseZstack', 'deg', 'Phase(Zstack)/deg', ''],
 423: ['<f4', 'abs_Zstack', 'Ohm', '|Zstack|/Ohm', ''],
 424: ['<f4', 'ReZstack', 'Ohm', 'Re(Zstack)/Ohm', ''],
 425: ['<f4', 'neg_ImZstack', 'Ohm', '-Im(Zstack)/Ohm', ''],
 426: ['<f4', 'avr_Estack', 'V', '<Estack>/V', ''],
 427: ['<f4', 'NN_427', '', '', ''],
 428: ['<f4', 'NN_428', '', '', ''],
 429: ['<f4', 'NN_429', '', '', ''],
 430: ['<f4', 'phaseDiffZ', 'deg', 'Phase(Zwe-ce)/deg', ''],
 431: ['<f4', 'absDiffZ', 'Ohm', '|Zwe-ce|/Ohm', ''],
 432: ['<f4', 'ReDiffZ', 'Ohm', 'Re(Zwe-ce)/Ohm', ''],
 433: ['<f4', 'neg_ImZwe_ce', 'Ohm', '-Im(Zwe-ce)/Ohm', ''],
 434: ['<f4', 'Z', 'W', '(Q-Qo)/C', ''],
 435: ['<f4', 'QminusQnull', 'C', 'dQ/C', ''],
 436: ['<f4', 'NN_436', '', '', ''],
 437: ['<f4', 'NN_437', '', '', ''],
 438: ['<f8', 'step_time', 's', 'step time/s', ''],
 439: ['<f4', 'NN_439', '', '', ''],
 440: ['<f4', 'NN_440', '', '', ''],
 441: ['<f4', 'avr_Ecv', 'V', '<Ecv>/V', ''],
 442: ['<f4', 'NN_442', '', '', ''],
 443: ['<f4', 'NN_443', '', '', ''],
 444: ['<f4', 'NN_444', '', '', ''],
 445: ['<f4', 'NN_445', '', '', ''],
 446: ['<f4', 'NN_446', '', '', ''],
 447: ['<f4', 'NN_447', '', '', ''],
 448: ['<f4', 'NN_448', '', '', ''],
 449: ['<f4', 'NN_449', '', '', ''],
 450: ['<f4', 'NN_450', '', '', ''],
 451: ['<f4', 'NN_451', '', '', ''],
 452: ['<f4', 'NN_452', '', '', ''],
 453: ['<f4', 'NN_453', '', '', ''],
 454: ['<f4', 'NN_454', '', '', ''],
 455: ['<f4', 'NN_455', '', '', ''],
 456: ['<f4', 'NN_456', '', '', ''],
 457: ['<f4', 'NN_457', '', '', ''],
 458: ['<f4', 'NN_458', '', '', ''],
 459: ['<f4', 'NN_459', '', '', ''],
 460: ['<f4', 'NN_460', '', '', ''],
 461: ['<f4', 'NN_461', '', '', ''],
 462: ['<f4', 'Temperature', 'deg_C', 'Temperature/C', ''],
 463: ['<f4', 'NN_463', '', '', ''],
 464: ['<f4', 'NN_464', '', '', ''],
 465: ['<f4', 'NN_465', '', '', ''],
 466: ['<f4', 'NN_466', '', '', ''],
 467: ['<f8', 'QChargeDischarge', 'mAh', 'Q charge/discharge/mA.h', 'charge_capacity_txt'],
 468: ['<u4', 'half_cycle', '', 'half cycle', 'cycle_index_txt'],
 469: ['<u4', 'z_cycle', '', 'z cycle', ''],
 470: ['<f4', 'NN_470', '', '', ''],
 471: ['<f4', 'avEce', 'V', '<Ece>/V', ''],
 472: ['<f4', 'NN_472', '', '', ''],
 473: ['<f4', 'THD_Ewe', 'percent', 'THD Ewe/%', ''],
 474: ['<f4', 'THD_I', 'percent', 'THD I/%', ''],
 475: ['<f4', 'NN_475', '', '', ''],
 476: ['<f4', 'NSD_Ewe', 'percent', 'NSD Ewe/%', ''],
 477: ['<f4', 'NSD_I', 'percent', 'NSD I/%', ''],
 478: ['<f4', 'NN_478', '', '', ''],
 479: ['<f4', 'NSR_Ewe', 'percent', 'NSR Ewe/%', ''],
 480: ['<f4', 'NSR_I', 'percent', 'NSR I/%', ''],
 481: ['<f4', 'NN_481', '', '', ''],
 482: ['<f4', 'NN_482', '', '', ''],
 483: ['<f4', 'NN_483', '', '', ''],
 484: ['<f4', 'NN_484', '', '', ''],
 485: ['<f4', 'NN_485', '', '', ''],
 486: ['<f4', 'abs_Ewe_h2', 'V', '|Ewe h2|/V', ''],
 487: ['<f4', 'abs_Ewe_h3', 'V', '|Ewe h3|/V', ''],
 488: ['<f4', 'abs_Ewe_h4', 'V', '|Ewe h4|/V', ''],
 489: ['<f4', 'abs_Ewe_h5', 'V', '|Ewe h5|/V', ''],
 490: ['<f4', 'abs_Ewe_h6', 'V', '|Ewe h6|/V', ''],
 491: ['<f4', 'abs_Ewe_h7', 'V', '|Ewe h7|/V', ''],
 492: ['<f4', 'abs_I_h2', 'A', '|I h2|/A', ''],
 493: ['<f4', 'abs_I_h3', 'A', '|I h3|/A', ''],
 494: ['<f4', 'abs_I_h4', 'A', '|I h4|/A', ''],
 495: ['<f4', 'abs_I_h5', 'A', '|I h5|/A', ''],
 496: ['<f4', 'abs_I_h6', 'A', '|I h6|/A', ''],
 497: ['<f4', 'abs_I_h7', 'A', '|I h7|/A', ''],
 498: ['<f8', 'Q_charge', 'mAh', 'Q charge/mA.h', 'charge_capacity_txt'],
 499: ['<f8', 'Q_discharge', 'mAh', 'Q discharge/mA.h', 'discharge_capacity_txt'],
 500: ['<f8', 'step_time', 's', 'step time/s', 'step_time_txt'],
 501: ['<f8', 'Efficiency', 'percent', 'Efficiency/%', ''],
 502: ['<f8', 'Capacity', 'mAh', 'Capacity/mA.h', 'charge_capacity_txt'],
 503: ['<f4', 'NN_503', '', '', ''],
 504: ['<f4', 'NN_504', '', '', ''],
 505: ['<f4', 'Rdc', 'Ohm', 'Rdc/Ohm', 'internal_resistance_txt'],
 506: ['<f4', 'NN_506', '', '', ''],
 507: ['<f4', 'NN_507', '', '', ''],
 508: ['<f4', 'NN_508', '', '', ''],
 509: ['<u1', 'Acir', 'Dcir_Control', 'Acir/Dcir Control', '']}

bl_flags: Dict[int, Tuple[str, int, type]] = {
    1: ("mode", 0x03, np.uint8),
    2: ("ox_red", 0x04, np.uint8),
    3: ("error", 0x08, np.uint8),
    21: ("control_changes", 0x10, np.uint8),
    31: ("Ns_changes", 0x20, np.uint8),
    65: ("counter_inc", 0x80, np.uint8),
}


def read_mpr_modules(file_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read all top-level modules from a BioLogic .mpr binary file."""
    import struct
    path = Path(file_path)
    total_size = path.stat().st_size
    modules: Dict[str, Dict[str, Any]] = {}

    with open(path, "rb") as f:
        label = f.read(len(mpr_label))
        if b"BIO-LOGIC" not in label and b"VMP" not in label:
            raise ValueError(f"Invalid BioLogic MPR binary magic header in {path.name}")

        while f.tell() < total_size:
            magic = f.read(len(b"MODULE"))
            if not magic or len(magic) < len(b"MODULE"):
                break
            if magic != b"MODULE":
                # Resynchronize to next module tag if byte alignment differs
                pos = f.tell() - len(b"MODULE")
                buf = f.read(8192)
                p = buf.find(b"MODULE")
                if p != -1:
                    f.seek(pos + p + len(b"MODULE"))
                else:
                    break

            shortname_raw = f.read(10)
            longname_raw = f.read(25)
            if len(shortname_raw) < 10 or len(longname_raw) < 25:
                break
            shortname = shortname_raw.decode("latin-1", errors="ignore").strip()
            longname = longname_raw.decode("latin-1", errors="ignore").strip()

            flag_bytes = f.read(4)
            if len(flag_bytes) < 4:
                break
            flag_u4 = struct.unpack("<I", flag_bytes)[0]
            if flag_u4 == 0xFFFFFFFF:
                # EC-Lab v11+ 64-bit extended length format
                ext_bytes = f.read(12)  # 8 bytes length + 4 bytes version
                if len(ext_bytes) < 12:
                    break
                length, version = struct.unpack("<QI", ext_bytes)
                date = f.read(8)
            else:
                ver_bytes = f.read(4)
                if len(ver_bytes) < 4:
                    break
                version = struct.unpack("<I", ver_bytes)[0]
                length = flag_u4
                date = f.read(8)

            data_bytes = f.read(length)
            hdr_dict = {
                "shortname": shortname_raw,
                "longname": longname_raw,
                "length": length,
                "version": version,
                "date": date,
                "data": data_bytes,
            }
            modules[shortname] = hdr_dict

    return modules


def decode_mpr_data_module(data_mod: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Decode the 'VMP data' module into a dictionary of named 1D numpy arrays."""
    import struct
    raw_bytes = data_mod["data"]
    version = int(data_mod.get("version", 0))

    if len(raw_bytes) < 6:
        raise ValueError("BioLogic MPR data module is empty or corrupted.")

    n_data_points = int(struct.unpack_from("<I", raw_bytes, 0)[0])
    if n_data_points <= 0:
        return {}

    if version == 0:
        n_columns = int(raw_bytes[4])
        col_types = list(raw_bytes[5 : 5 + n_columns])
        min_header_len = 5 + n_columns
        default_offset = 100
    elif raw_bytes[5] == 0 and 1 <= int(struct.unpack_from("<H", raw_bytes, 4)[0]) <= 100 and (6 + 2 * int(struct.unpack_from("<H", raw_bytes, 4)[0])) <= len(raw_bytes):
        # EC-Lab v11+ 2-byte column count; col_types start at offset 6
        n_columns = int(struct.unpack_from("<H", raw_bytes, 4)[0])
        col_types = [struct.unpack_from("<H", raw_bytes, 6 + 2 * i)[0] for i in range(n_columns)]
        min_header_len = 6 + (n_columns * 2)
        default_offset = 405
    else:
        # EC-Lab v2/v3 1-byte column count; col_types start at offset 5
        n_columns = int(raw_bytes[4])
        col_types = [struct.unpack_from("<H", raw_bytes, 5 + 2 * i)[0] for i in range(n_columns)]
        min_header_len = 5 + (n_columns * 2)
        default_offset = 405

    if n_columns <= 0:
        return {}

    dtype_fields = []
    flags_present = False
    flag_keys = {1, 2, 3, 21, 31, 65}
    active_flag_cols: List[int] = []
    seen_field_names = set()

    for col_id in col_types:
        cid = int(col_id)
        if cid in flag_keys:
            active_flag_cols.append(cid)
            if not flags_present:
                dtype_fields.append(("flags", "u1"))
                seen_field_names.add("flags")
                flags_present = True
            continue

        if cid in bl_dtypes:
            dt_str, name, unit, col_name, *rest = bl_dtypes[cid]
            field_name = col_name if col_name else f"col_{cid}"
        else:
            dt_str = "<f4"
            field_name = f"col_{cid}"

        clean_field_name = field_name.encode("ascii", errors="ignore").decode()
        fld = clean_field_name
        idx = 2
        while fld in seen_field_names:
            fld = f"{clean_field_name}_{idx}"
            idx += 1
        seen_field_names.add(fld)
        dtype_fields.append((fld, dt_str))

    struct_dtype = np.dtype(dtype_fields)
    expected_row_size = struct_dtype.itemsize

    # Dynamic header size resolution:
    calc_offset = len(raw_bytes) - (n_data_points * expected_row_size)
    if min_header_len <= calc_offset <= 8192:
        data_offset = calc_offset
    else:
        data_offset = default_offset

    main_data = raw_bytes[data_offset:]

    max_pts = len(main_data) // expected_row_size if expected_row_size > 0 else 0
    actual_pts = min(n_data_points, max_pts)

    if actual_pts <= 0:
        return {}

    parsed_struct = np.frombuffer(main_data[: actual_pts * expected_row_size], dtype=struct_dtype)

    result: Dict[str, np.ndarray] = {}
    for name in parsed_struct.dtype.names:
        result[name] = parsed_struct[name]

    if "flags" in result and active_flag_cols:
        flag_byte = result["flags"]
        for cid in active_flag_cols:
            if cid in bl_flags:
                flag_name, bitmask, target_dtype = bl_flags[cid]
                result[flag_name] = ((flag_byte & bitmask) > 0).astype(int)

    return result
