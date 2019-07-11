/* automatically generated by rust-bindgen */

pub const MKL_DOMAIN_ALL : u32 = 0;
pub const MKL_DOMAIN_BLAS : u32 = 1;
pub const MKL_DOMAIN_FFT : u32 = 2;
pub const MKL_DOMAIN_VML : u32 = 3;
pub const MKL_DOMAIN_PARDISO : u32 = 4;
pub const MKL_CBWR_BRANCH : u32 = 1;
pub const MKL_CBWR_ALL : i32 = -1;
pub const MKL_CBWR_UNSET_ALL : u32 = 0;
pub const MKL_CBWR_OFF : u32 = 0;
pub const MKL_CBWR_BRANCH_OFF : u32 = 1;
pub const MKL_CBWR_AUTO : u32 = 2;
pub const MKL_CBWR_COMPATIBLE : u32 = 3;
pub const MKL_CBWR_SSE2 : u32 = 4;
pub const MKL_CBWR_SSSE3 : u32 = 6;
pub const MKL_CBWR_SSE4_1 : u32 = 7;
pub const MKL_CBWR_SSE4_2 : u32 = 8;
pub const MKL_CBWR_AVX : u32 = 9;
pub const MKL_CBWR_AVX2 : u32 = 10;
pub const MKL_CBWR_AVX512_MIC : u32 = 11;
pub const MKL_CBWR_AVX512 : u32 = 12;
pub const MKL_CBWR_AVX512_MIC_E1 : u32 = 13;
pub const MKL_CBWR_SUCCESS : u32 = 0;
pub const MKL_CBWR_ERR_INVALID_SETTINGS : i32 = -1;
pub const MKL_CBWR_ERR_INVALID_INPUT : i32 = -2;
pub const MKL_CBWR_ERR_UNSUPPORTED_BRANCH : i32 = -3;
pub const MKL_CBWR_ERR_UNKNOWN_BRANCH : i32 = -4;
pub const MKL_CBWR_ERR_MODE_CHANGE_FAILURE : i32 = -8;
pub const MKL_CBWR_SSE3 : u32 = 5;
# [ repr ( C ) ] # [ derive ( Debug , Copy , Clone ) ] pub struct _MKL_Complex8 { pub real : f32 , pub imag : f32 , } # [ test ] fn bindgen_test_layout__MKL_Complex8 ( ) { assert_eq ! ( :: std :: mem :: size_of :: < _MKL_Complex8 > ( ) , 8usize , concat ! ( "Size of: " , stringify ! ( _MKL_Complex8 ) ) );
assert_eq ! ( :: std :: mem :: align_of :: < _MKL_Complex8 > ( ) , 4usize , concat ! ( "Alignment of " , stringify ! ( _MKL_Complex8 ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < _MKL_Complex8 > ( ) ) ) . real as * const _ as usize } , 0usize , concat ! ( "Offset of field: " , stringify ! ( _MKL_Complex8 ) , "::" , stringify ! ( real ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < _MKL_Complex8 > ( ) ) ) . imag as * const _ as usize } , 4usize , concat ! ( "Offset of field: " , stringify ! ( _MKL_Complex8 ) , "::" , stringify ! ( imag ) ) );
} pub type MKL_Complex8 = _MKL_Complex8;
# [ repr ( C ) ] # [ derive ( Debug , Copy , Clone ) ] pub struct _MKL_Complex16 { pub real : f64 , pub imag : f64 , } # [ test ] fn bindgen_test_layout__MKL_Complex16 ( ) { assert_eq ! ( :: std :: mem :: size_of :: < _MKL_Complex16 > ( ) , 16usize , concat ! ( "Size of: " , stringify ! ( _MKL_Complex16 ) ) );
assert_eq ! ( :: std :: mem :: align_of :: < _MKL_Complex16 > ( ) , 8usize , concat ! ( "Alignment of " , stringify ! ( _MKL_Complex16 ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < _MKL_Complex16 > ( ) ) ) . real as * const _ as usize } , 0usize , concat ! ( "Offset of field: " , stringify ! ( _MKL_Complex16 ) , "::" , stringify ! ( real ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < _MKL_Complex16 > ( ) ) ) . imag as * const _ as usize } , 8usize , concat ! ( "Offset of field: " , stringify ! ( _MKL_Complex16 ) , "::" , stringify ! ( imag ) ) );
} pub type MKL_Complex16 = _MKL_Complex16;
# [ repr ( C ) ] # [ derive ( Debug , Copy , Clone ) ] pub struct MKLVersion { pub MajorVersion : :: std :: os :: raw :: c_int , pub MinorVersion : :: std :: os :: raw :: c_int , pub UpdateVersion : :: std :: os :: raw :: c_int , pub ProductStatus : * mut :: std :: os :: raw :: c_char , pub Build : * mut :: std :: os :: raw :: c_char , pub Processor : * mut :: std :: os :: raw :: c_char , pub Platform : * mut :: std :: os :: raw :: c_char , } # [ test ] fn bindgen_test_layout_MKLVersion ( ) { assert_eq ! ( :: std :: mem :: size_of :: < MKLVersion > ( ) , 48usize , concat ! ( "Size of: " , stringify ! ( MKLVersion ) ) );
assert_eq ! ( :: std :: mem :: align_of :: < MKLVersion > ( ) , 8usize , concat ! ( "Alignment of " , stringify ! ( MKLVersion ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < MKLVersion > ( ) ) ) . MajorVersion as * const _ as usize } , 0usize , concat ! ( "Offset of field: " , stringify ! ( MKLVersion ) , "::" , stringify ! ( MajorVersion ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < MKLVersion > ( ) ) ) . MinorVersion as * const _ as usize } , 4usize , concat ! ( "Offset of field: " , stringify ! ( MKLVersion ) , "::" , stringify ! ( MinorVersion ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < MKLVersion > ( ) ) ) . UpdateVersion as * const _ as usize } , 8usize , concat ! ( "Offset of field: " , stringify ! ( MKLVersion ) , "::" , stringify ! ( UpdateVersion ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < MKLVersion > ( ) ) ) . ProductStatus as * const _ as usize } , 16usize , concat ! ( "Offset of field: " , stringify ! ( MKLVersion ) , "::" , stringify ! ( ProductStatus ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < MKLVersion > ( ) ) ) . Build as * const _ as usize } , 24usize , concat ! ( "Offset of field: " , stringify ! ( MKLVersion ) , "::" , stringify ! ( Build ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < MKLVersion > ( ) ) ) . Processor as * const _ as usize } , 32usize , concat ! ( "Offset of field: " , stringify ! ( MKLVersion ) , "::" , stringify ! ( Processor ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < MKLVersion > ( ) ) ) . Platform as * const _ as usize } , 40usize , concat ! ( "Offset of field: " , stringify ! ( MKLVersion ) , "::" , stringify ! ( Platform ) ) );
} pub const MKL_LAYOUT_MKL_ROW_MAJOR : MKL_LAYOUT = 101;
pub const MKL_LAYOUT_MKL_COL_MAJOR : MKL_LAYOUT = 102;
pub type MKL_LAYOUT = u32;
pub const MKL_TRANSPOSE_MKL_NOTRANS : MKL_TRANSPOSE = 111;
pub const MKL_TRANSPOSE_MKL_TRANS : MKL_TRANSPOSE = 112;
pub const MKL_TRANSPOSE_MKL_CONJTRANS : MKL_TRANSPOSE = 113;
pub type MKL_TRANSPOSE = u32;
pub const MKL_UPLO_MKL_UPPER : MKL_UPLO = 121;
pub const MKL_UPLO_MKL_LOWER : MKL_UPLO = 122;
pub type MKL_UPLO = u32;
pub const MKL_DIAG_MKL_NONUNIT : MKL_DIAG = 131;
pub const MKL_DIAG_MKL_UNIT : MKL_DIAG = 132;
pub type MKL_DIAG = u32;
pub const MKL_SIDE_MKL_LEFT : MKL_SIDE = 141;
pub const MKL_SIDE_MKL_RIGHT : MKL_SIDE = 142;
pub type MKL_SIDE = u32;
pub const MKL_COMPACT_PACK_MKL_COMPACT_SSE : MKL_COMPACT_PACK = 181;
pub const MKL_COMPACT_PACK_MKL_COMPACT_AVX : MKL_COMPACT_PACK = 182;
pub const MKL_COMPACT_PACK_MKL_COMPACT_AVX512 : MKL_COMPACT_PACK = 183;
pub type MKL_COMPACT_PACK = u32;
pub type DFTaskPtr = * mut :: std :: os :: raw :: c_void;
# [ repr ( C ) ] # [ derive ( Debug , Copy , Clone ) ] pub struct _dfSearchCallBackLibraryParams { pub limit_type_flag : :: std :: os :: raw :: c_int , } # [ test ] fn bindgen_test_layout__dfSearchCallBackLibraryParams ( ) { assert_eq ! ( :: std :: mem :: size_of :: < _dfSearchCallBackLibraryParams > ( ) , 4usize , concat ! ( "Size of: " , stringify ! ( _dfSearchCallBackLibraryParams ) ) );
assert_eq ! ( :: std :: mem :: align_of :: < _dfSearchCallBackLibraryParams > ( ) , 4usize , concat ! ( "Alignment of " , stringify ! ( _dfSearchCallBackLibraryParams ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < _dfSearchCallBackLibraryParams > ( ) ) ) . limit_type_flag as * const _ as usize } , 0usize , concat ! ( "Offset of field: " , stringify ! ( _dfSearchCallBackLibraryParams ) , "::" , stringify ! ( limit_type_flag ) ) );
} pub type dfSearchCallBackLibraryParams = _dfSearchCallBackLibraryParams;
# [ repr ( C ) ] # [ derive ( Debug , Copy , Clone ) ] pub struct _dfInterpCallBackLibraryParams { pub reserved1 : :: std :: os :: raw :: c_int , } # [ test ] fn bindgen_test_layout__dfInterpCallBackLibraryParams ( ) { assert_eq ! ( :: std :: mem :: size_of :: < _dfInterpCallBackLibraryParams > ( ) , 4usize , concat ! ( "Size of: " , stringify ! ( _dfInterpCallBackLibraryParams ) ) );
assert_eq ! ( :: std :: mem :: align_of :: < _dfInterpCallBackLibraryParams > ( ) , 4usize , concat ! ( "Alignment of " , stringify ! ( _dfInterpCallBackLibraryParams ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < _dfInterpCallBackLibraryParams > ( ) ) ) . reserved1 as * const _ as usize } , 0usize , concat ! ( "Offset of field: " , stringify ! ( _dfInterpCallBackLibraryParams ) , "::" , stringify ! ( reserved1 ) ) );
} pub type dfInterpCallBackLibraryParams = _dfInterpCallBackLibraryParams;
# [ repr ( C ) ] # [ derive ( Debug , Copy , Clone ) ] pub struct _dfIntegrCallBackLibraryParams { pub reserved1 : :: std :: os :: raw :: c_int , } # [ test ] fn bindgen_test_layout__dfIntegrCallBackLibraryParams ( ) { assert_eq ! ( :: std :: mem :: size_of :: < _dfIntegrCallBackLibraryParams > ( ) , 4usize , concat ! ( "Size of: " , stringify ! ( _dfIntegrCallBackLibraryParams ) ) );
assert_eq ! ( :: std :: mem :: align_of :: < _dfIntegrCallBackLibraryParams > ( ) , 4usize , concat ! ( "Alignment of " , stringify ! ( _dfIntegrCallBackLibraryParams ) ) );
assert_eq ! ( unsafe { & ( * ( :: std :: ptr :: null :: < _dfIntegrCallBackLibraryParams > ( ) ) ) . reserved1 as * const _ as usize } , 0usize , concat ! ( "Offset of field: " , stringify ! ( _dfIntegrCallBackLibraryParams ) , "::" , stringify ! ( reserved1 ) ) );
} pub type dfIntegrCallBackLibraryParams = _dfIntegrCallBackLibraryParams;
pub type dfsInterpCallBack = :: std :: option :: Option < unsafe extern "C" fn ( n : * mut :: std :: os :: raw :: c_longlong , cell : * mut :: std :: os :: raw :: c_longlong , site : * mut f32 , r : * mut f32 , user_param : * mut :: std :: os :: raw :: c_void , library_params : * mut dfInterpCallBackLibraryParams ) -> :: std :: os :: raw :: c_int >;
pub type dfdInterpCallBack = :: std :: option :: Option < unsafe extern "C" fn ( n : * mut :: std :: os :: raw :: c_longlong , cell : * mut :: std :: os :: raw :: c_longlong , site : * mut f64 , r : * mut f64 , user_param : * mut :: std :: os :: raw :: c_void , library_params : * mut dfInterpCallBackLibraryParams ) -> :: std :: os :: raw :: c_int >;
pub type dfsIntegrCallBack = :: std :: option :: Option < unsafe extern "C" fn ( n : * mut :: std :: os :: raw :: c_longlong , lcell : * mut :: std :: os :: raw :: c_longlong , llim : * mut f32 , rcell : * mut :: std :: os :: raw :: c_longlong , rlim : * mut f32 , r : * mut f32 , user_params : * mut :: std :: os :: raw :: c_void , library_params : * mut dfIntegrCallBackLibraryParams ) -> :: std :: os :: raw :: c_int >;
pub type dfdIntegrCallBack = :: std :: option :: Option < unsafe extern "C" fn ( n : * mut :: std :: os :: raw :: c_longlong , lcell : * mut :: std :: os :: raw :: c_longlong , llim : * mut f64 , rcell : * mut :: std :: os :: raw :: c_longlong , rlim : * mut f64 , r : * mut f64 , user_params : * mut :: std :: os :: raw :: c_void , library_params : * mut dfIntegrCallBackLibraryParams ) -> :: std :: os :: raw :: c_int >;
pub type dfsSearchCellsCallBack = :: std :: option :: Option < unsafe extern "C" fn ( n : * mut :: std :: os :: raw :: c_longlong , site : * mut f32 , cell : * mut :: std :: os :: raw :: c_longlong , flag : * mut :: std :: os :: raw :: c_int , user_params : * mut :: std :: os :: raw :: c_void , library_params : * mut dfSearchCallBackLibraryParams ) -> :: std :: os :: raw :: c_int >;
pub type dfdSearchCellsCallBack = :: std :: option :: Option < unsafe extern "C" fn ( n : * mut :: std :: os :: raw :: c_longlong , site : * mut f64 , cell : * mut :: std :: os :: raw :: c_longlong , flag : * mut :: std :: os :: raw :: c_int , user_params : * mut :: std :: os :: raw :: c_void , library_params : * mut dfSearchCallBackLibraryParams ) -> :: std :: os :: raw :: c_int > ;