require 'set'

module Hooks
    module ClassMethods
        def define_hook(name)
            name = name.to_sym
            @hook_names << name
            instance_eval %Q{
                def #{name}(&block)
                    @hook_callbacks[:#{name}] << block
                end
                }
        end

        def __initialize
            @hook_names ||= Set.new
            @hook_callbacks ||= Hash.new do |h,k|
                h[k] = Array.new
            end
        end

        def callbacks_for_hook(name)
            name = name.to_sym
            raise ArgumentError,"hook #{name} is not known" unless hook?(name)
            if @hook_callbacks.has_key?(name)
                @hook_callbacks[name]
            else
                []
            end
        end

        def hook?(name)
            name = name.to_sym
            if @hook_names.include?(name)
                true
            else
                false
            end
        end

        def callback?(name)
            !callbacks_for_hook(name).empty?
        end

        def run_hook(name,*args)
            callbacks = callbacks_for_hook(name)
            callbacks.map do |c|
                c.call(*args)
            end
        end
    end

    def self.included(base)
        base.extend ClassMethods
        base.__initialize
    end
end
