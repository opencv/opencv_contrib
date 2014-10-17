begin
    require 'ffi'
rescue LoadError
    STDERR.puts "Cannot require 'ffi'"
    STDERR.puts "You can install ffi via 'gem install ffi'"
    exit 1
end
require File.join(File.dirname(__FILE__),'<%= file_prefix %>_types.rb')
<%= required_module_names %>

<%= add_doc -%>
module <%= name %>
    # low level accessors the wrapped library 
    module Rbind
        extend FFI::Library

        #load library <%= library_name %>
        path = File.dirname(__FILE__)
        path = if Dir.exist?(path)
                   Dir.chdir(path) do
                       path = Dir.glob("lib<%= library_name %>.*").first
                       File.absolute_path(path) if path
                   end
               end
        ffi_lib ["<%= library_name %>", path]

        # @!group Enums
        <%= add_enums%>
        # @!endgroup

        #add error checking
        #rbindCreateMatrix -> create_matrix
        def self.attach_function(ruby_name,c_name, args, returns,error_checking=true)
            return super(ruby_name,c_name,args,returns) unless error_checking || !returns

            #add accessor for c function
            super("orig_#{ruby_name}", c_name, args, returns)

            #add ruby method that does error checking after the c
            #function was called
            line_no = __LINE__; str = %Q{
                        def #{ruby_name}(*args, &block)
                            val = orig_#{ruby_name}(*args,&block)
                            if has_error
                                except = RuntimeError.new(last_error)
                                except.set_backtrace(caller)
                                clear_error
                                raise except
                            end
                            val
                        rescue
                            $@.delete_if{|s| %r"#{Regexp.quote(__FILE__)}"o =~ s}
                            Kernel.raise
                        end
            }
            instance_eval(str,__FILE__,line_no)
            self
        end

        # add basic rbind functions which must not get error checking
        attach_function :has_error,:rbindHasError,[],:bool,false
        attach_function :last_error,:rbindGetLastError,[],:string,false
        attach_function :clear_error,:rbindClearError,[],:void,false

        # add accessor for wrapped methods
        <%= add_accessors %>
    end
end
