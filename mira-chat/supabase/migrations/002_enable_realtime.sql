-- Enable Supabase Realtime for key tables
ALTER PUBLICATION supabase_realtime ADD TABLE public.events;
ALTER PUBLICATION supabase_realtime ADD TABLE public.object_requests;
ALTER PUBLICATION supabase_realtime ADD TABLE public.object_state;
