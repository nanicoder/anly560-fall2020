SELECT first_name, last_name, email 
FROM customer
where email > 'w'
order by email;
##List all the email greater than 'w', where 'wa' is greater value than 'w'

##Lower() method usage
select lower(title), title
from film;

##sounddex() method usage : 'soundex' method converts any string value into a hexa decimal value
select soundex(title), title 
from film;

## agv(), count(), sum() methods
select count(title) from film;
select count(*) from film;
select avg(rental_rate) average_rental, avg(replacement_cost) average_replacement from film;
## calculating the difference values
select title, (replacement_cost - rental_rate) differnece from film;
select avg(replacement_cost - rental_rate) differnece from film;

##Grouping, distinct  
select distinct rating from film order by rating;
select rating, count(*) from film group by rating;

# filter PRIOR to grouping
select rating, count(*) from film where rating in ('G', 'PG', 'PG-13') group by rating;

# filter after grouping
select rating, count(*) tot_count from film group by rating having tot_count > 200;
select rating, count(*) tot_count from film where rating in ('G', 'PG', 'PG-13') group by rating having tot_count > 200;

select * from employee where first_name = 'Puddi';

select * FROM actor where first_name = 'Rama';
DELETE FROM actor where first_name = 'Rama';