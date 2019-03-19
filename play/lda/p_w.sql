-- for each token in a document sample from this
select 
	word_id, 
	word_str, 
	sum(p_wz) as p_W
from (
	select 
		topic_id,  
		word_id, 
		round((topic_weight * word_p), 8) as p_wz
	from doctopic dt
		join topicword_v tw
			using (topic_id)
	where doc_id = 10
	order by topic_id, p_wz desc
)
join word using(word_id)
group by word_id
order by p_W  desc
limit 20