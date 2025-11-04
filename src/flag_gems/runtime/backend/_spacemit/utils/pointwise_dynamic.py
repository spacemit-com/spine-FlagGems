from flag_gems.utils.pointwise_dynamic import (
    ModuleGenerator as BaseModuleGenerator,
    KernelGenerator as BaseKernelGenerator,
    WrapperGenerator,
)

class KernelGenerator(BaseKernelGenerator):
    def gen_body_gsl_with_bptr(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in smt.parallel(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_with_bptr(code)

    def gen_body_gsl_without_bptr(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in smt.parallel(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_without_bptr(code)

    def gen_body_gsl_1d_tile(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in smt.parallel(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_1d_tile(code)

class SpacemitModuleGenerator(BaseModuleGenerator):
    def __init__(self, function_schema, scalar_fn, ndim, jit_fn_name, wrapper_name, config):
        self.config = config
        self.wrapper_gen = WrapperGenerator(
            function_schema, jit_fn_name, ndim, wrapper_name, config
        )
        self.kernel_gen = KernelGenerator(
            function_schema, scalar_fn, ndim, jit_fn_name, config
        )
        self.jit_fn_name = jit_fn_name

    @staticmethod
    def generate_imports(code):
        BaseModuleGenerator.generate_imports(code)
        code.writeline("import triton.language.extra.smt as smt")
        code.newline()
        return code