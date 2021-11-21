use nalgebra as na;
use std::ffi::CString;
pub mod gl;

pub struct Program {
    id: u32,
    gl: gl::Gl,
    pub model: na::Matrix4<f32>,
    model_location: i32,
    pub view: na::Matrix4<f32>,
    view_location: i32,
    pub proj: na::Matrix4<f32>,
    proj_location: i32,
}

impl Program {
    pub fn new(gl: gl::Gl) -> Program {
        let vertex_shader = unsafe { gl.CreateShader(gl::VERTEX_SHADER) };
        unsafe {
            let vertex_code = CString::new(include_str!("default.vert"))
                .expect("Vertex shader source has a null terminator");
            gl.ShaderSource(
                vertex_shader,
                1,
                &vertex_code.as_ptr(),
                &[vertex_code.as_bytes_with_nul().len() as i32] as *const i32,
            );
            gl.CompileShader(vertex_shader);
            let mut vertex_info_log = Vec::<u8>::with_capacity(2048);
            let mut vertex_info_log_len: i32 = 0;
            gl.GetShaderInfoLog(
                vertex_shader,
                vertex_info_log.capacity() as i32,
                &mut vertex_info_log_len as *mut i32,
                vertex_info_log.as_mut_ptr().cast(),
            );
            vertex_info_log.set_len(vertex_info_log_len as usize);
            println!(
                "len of info log: {}, len of buffer {}",
                vertex_info_log_len,
                vertex_info_log.len()
            );
            let vertex_info_log_str =
                String::from_utf8(vertex_info_log).expect("Info log is not utf8");
            println!("Vertex Shader Log:\n {}", vertex_info_log_str);
        }

        let fragment_shader = unsafe { gl.CreateShader(gl::FRAGMENT_SHADER) };
        unsafe {
            let fragment_code = CString::new(include_str!("default.frag"))
                .expect("Vertex shader source has a null terminator");
            gl.ShaderSource(
                fragment_shader,
                1,
                &fragment_code.as_ptr(),
                &[fragment_code.as_bytes_with_nul().len() as i32] as *const i32,
            );
            gl.CompileShader(fragment_shader);
            let mut fragment_info_log = Vec::<u8>::with_capacity(2048);
            let mut fragment_info_log_len: i32 = 0;
            gl.GetShaderInfoLog(
                fragment_shader,
                fragment_info_log.capacity() as i32,
                &mut fragment_info_log_len,
                fragment_info_log.as_mut_ptr().cast(),
            );
            fragment_info_log.set_len(fragment_info_log_len as usize);
            println!(
                "len of info log: {}, len of buffer {}",
                fragment_info_log_len,
                fragment_info_log.len()
            );
            let fragment_info_log_str =
                String::from_utf8(fragment_info_log).expect("Info log is not utf8");
            println!("Fragment Shader Log:\n {}", fragment_info_log_str);
        }

        let program_id = unsafe { gl.CreateProgram() };
        unsafe {
            gl.AttachShader(program_id, vertex_shader);
            gl.AttachShader(program_id, fragment_shader);
            gl.LinkProgram(program_id);
            let mut link_status: i32 = 0;
            gl.GetProgramiv(program_id, gl::LINK_STATUS, &mut link_status);
            if link_status as u8 == gl::TRUE {
                println!("Linking was successful");
            }
            if link_status as u8 == gl::FALSE {
                println!("Linking failed");
            }
        }
        let model_location =
            unsafe { gl.GetUniformLocation(program_id, b"model\0".as_ptr().cast()) };
        println!("Uniform model location: {}", model_location);
        let view_location = unsafe { gl.GetUniformLocation(program_id, b"view\0".as_ptr().cast()) };
        println!("Uniform view location: {}", view_location);
        let proj_location = unsafe { gl.GetUniformLocation(program_id, b"proj\0".as_ptr().cast()) };
        println!("Uniform proj location: {}", proj_location);
        Program {
            gl,
            id: program_id,
            model_location,
            model: na::Matrix4::<f32>::identity(),
            view_location,
            view: na::Matrix4::<f32>::identity(),
            proj_location,
            proj: na::Matrix4::<f32>::identity(),
        }
    }

    pub fn bind(&self) {
        unsafe {
            self.gl.UseProgram(self.id);
            self.gl
                .UniformMatrix4fv(self.model_location, 1, gl::FALSE, self.model.as_ptr());
            self.gl
                .UniformMatrix4fv(self.proj_location, 1, gl::FALSE, self.proj.as_ptr());
            self.gl
                .UniformMatrix4fv(self.view_location, 1, gl::FALSE, self.view.as_ptr());
            let error_code = self.gl.GetError();
            match error_code {
                gl::NO_ERROR => (),
                gl::INVALID_ENUM => println!("Invalid enum OpenGL error while binding program"),
                gl::INVALID_VALUE => println!("Invalid value OpenGL error while binding program"),
                gl::INVALID_OPERATION => {
                    println!("Invalid operation OpenGL error while binding program");
                }
                _ => println!("Unkown OpenGL error while binding program"),
            }
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct Vertex {
    pub vertex: [f32; 2],
    pub color: [f32; 3],
}

pub struct VertexArray {
    gl: gl::Gl,
    vao: u32,
    vbo: u32,
}

impl VertexArray {
    #[allow(clippy::similar_names)]
    pub fn new(gl: gl::Gl) -> VertexArray {
        let mut vao: u32 = 0;
        let mut vbo: u32 = 0;
        unsafe {
            gl.GenVertexArrays(1, &mut vao);
            gl.GenBuffers(1, &mut vbo);
            gl.BindVertexArray(vao);
            gl.EnableVertexAttribArray(0);
            gl.EnableVertexAttribArray(1);
            gl.BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl.VertexAttribPointer(
                0,
                2,
                gl::FLOAT,
                gl::FALSE,
                5 * std::mem::size_of::<f32>() as i32,
                std::ptr::null(),
            );
            gl.VertexAttribPointer(
                1,
                3,
                gl::FLOAT,
                gl::FALSE,
                5 * std::mem::size_of::<f32>() as i32,
                (2 * std::mem::size_of::<f32>()) as *const std::ffi::c_void,
            );
            if gl.GetError() != gl::NO_ERROR {
                println!("OpenGL error occured");
            }
            gl.BindBuffer(gl::ARRAY_BUFFER, 0);
            gl.BindVertexArray(0);
        }
        println!("Vertex Array Object {}", vao);
        println!("Vertex Buffer Object {}", vbo);

        VertexArray { gl, vao, vbo }
    }

    pub fn with_vertecies(gl: gl::Gl, vertecies: &[Vertex]) -> VertexArray {
        let mut array = Self::new(gl);
        array.store(vertecies);
        array
    }

    pub fn store(&mut self, vertecies: &[Vertex]) {
        unsafe {
            self.gl.BindBuffer(gl::ARRAY_BUFFER, self.vbo);
            self.gl.BufferData(
                gl::ARRAY_BUFFER,
                (vertecies.len() * std::mem::size_of::<Vertex>()) as isize,
                vertecies.as_ptr().cast(),
                gl::STATIC_DRAW,
            );
            self.gl.BindBuffer(gl::ARRAY_BUFFER, 0);
            if self.gl.GetError() != gl::NO_ERROR {
                println!("OpenGL error occured while storing vertex data");
            }
        }
    }

    pub fn draw(&self, vertex_count: i32) {
        unsafe {
            self.gl.BindVertexArray(self.vao);
            self.gl.DrawArrays(gl::TRIANGLES, 0, vertex_count);
            self.gl.BindVertexArray(0);
            let error_code = self.gl.GetError();
            match error_code {
                gl::NO_ERROR => (),
                gl::INVALID_ENUM => println!("Invalid enum OpenGL error while drawing vao"),
                gl::INVALID_VALUE => println!("Invalid value OpenGL error while drawing vao"),
                gl::INVALID_OPERATION => {
                    println!("Invalid operation OpenGL error while drawing vao");
                }
                _ => println!("Unkown OpenGL error while drawing vao"),
            }
        }
    }
}
