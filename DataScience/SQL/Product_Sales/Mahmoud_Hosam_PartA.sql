
/* question Part A  number a   */
select * from TRANS where DateSold is null
delete from TRANS where DateSold is null





	/* question Part A  number b   */
	select WorkID , Title, WORK.Medium , WORK.ArtistID ,CONCAT(TRIM (FirstName) , ' ' ,TRIM (LastName) ) as FullName  from WORK

	inner join ARTIST on
	ARTIST.ArtistID = WORK.ArtistID

	where title LIKE '%Yellow%'
	  or title LIKE '%Blue%'
	  or title LIKE '%White%'






	/* question Part A  number C   */
	select YEAR(DateSold) , ARTIST.ArtistID ,  sum(SalesPrice) as SumOfSubTotal , avg(SalesPrice) as AverageOfSubtotal  from TRANS

	inner join WORK
	on WORK.WorkID = TRANS.WorkID

	inner join ARTIST 
	on ARTIST.ArtistID =WORK.ArtistID

	group by  YEAR(DateSold), ARTIST.ArtistID







	/* question Part A  number d   */
	select  ARTIST.ArtistID , FirstName , LastName , WORK.WorkID , Title   from TRANS

	inner join WORK
	on WORK.WorkID = TRANS.WorkID

	inner join ARTIST 
	on ARTIST.ArtistID =WORK.ArtistID

	where TRANS.SalesPrice> (select AVG(SalesPrice) from TRANS)







	/* question Part A  number E   */
	UPDATE CUSTOMER set
	EmailAddress = 'Johnson.lynda@somewhere.com' ,
	EncryptedPassword = 'aax1xbB'
	where FirstName='Lynda' and LastName = 'Johnson'







	/* question Part A  number f   */
	select * from (

		select Customer.* ,
		(DATEDIFF( day,LEAD(DateSold,1) over (PARTITION BY TRANS.CustomerID  ORDER BY DateSold DESC ) , DateSold  ))  as Days_Difference
		from TRANS 

		inner join CUSTOMER on 
		CUSTOMER.CustomerID =TRANS.CustomerID

	) as My_Table
	where My_Table.Days_Difference is not null





	 
	/* question Part A  number g   */
	 go
	 Create view CustomerTransactionSummaryView
	 as
	 (
	select concat(Trim(FirstName) ,' ', trim(LastName)) as FullName ,
		   Title , DateAcquired , DateSold ,
		  (SalesPrice -  AcquisitionPrice ) as Profit   from TRANS

	inner join WORK
	on WORK.WorkID = TRANS.WorkID

	inner join ARTIST 
	on ARTIST.ArtistID =WORK.ArtistID

	where TRANS.AskingPrice> 20000
	ORDER BY TRANS.AskingPrice  OFFSET 0 ROWS
	)

	 go
	select * from [CustomerTransactionSummaryView]




	/* question Part A  number h   */
	go
	WITH CTEPurchase(CustomerID,MinAcquisitionDate,MaxAcquisitionDate)  

	AS    
		(SELECT 
		CustomerID,
		MIN(TRANS.DateAcquired) AS MinAcquisitionDate,
		MAX(TRANS.DateAcquired) AS MaxAcquisitionDate 

		FROM  TRANS
		GROUP BY TRANS.CustomerID)
 
	select
	TransactionID,
	DateAcquired,
	CTEPurchase.CustomerID,
	Customer.LastName,
	Customer.FirstName,
	MinAcquisitionDate,
	MaxAcquisitionDate,

	CASE
	WHEN Medium='High Quality Limited Print' THEN 1
	WHEN Medium ='Color Aquatint' THEN 2
	WHEN Medium ='Water Color and Ink' THEN 3
	WHEN Medium ='Oil and Collage' THEN 4
	ELSE 5 END AS Medium

	into #Purchase

	FROM CTEPurchase 

	INNER JOIN CUSTOMER ON
	CTEPurchase.CustomerID = CUSTOMER.CustomerID

	INNER JOIN TRANS ON
	TRANS.CustomerID = CTEPurchase.CustomerID 

	INNER JOIN WORK ON
	TRANS.WorkID = WORK.WorkID

	where Year(DateAcquired)>=2015 AND Year(DateAcquired)<=2017


	select * from #Purchase